#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"

namespace duckdb {

namespace {

//! Hands the caching operator's Execute a prepared sequence of chunks, one per call
class EmitPreparedChunks : public CachingPhysicalOperator {
public:
	EmitPreparedChunks(PhysicalPlan &physical_plan, vector<LogicalType> types, vector<unique_ptr<DataChunk>> &chunks)
	    : CachingPhysicalOperator(physical_plan, PhysicalOperatorType::FILTER, std::move(types), 0), chunks(chunks) {
	}

	vector<unique_ptr<DataChunk>> &chunks;
	mutable idx_t next = 0;

	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override {
		return make_uniq<CachingOperatorState>();
	}

protected:
	OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                   GlobalOperatorState &gstate, OperatorState &state) const override {
		chunk.Reference(*chunks[next++]);
		return OperatorResultType::NEED_MORE_INPUT;
	}
};

unique_ptr<DataChunk> DictionaryChunk(Allocator &allocator, const buffer_ptr<DictionaryEntry> &strings,
                                      const buffer_ptr<DictionaryEntry> &numbers, const vector<sel_t> &indices) {
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(allocator, {LogicalType::VARCHAR, LogicalType::INTEGER});
	SelectionVector sel(indices.size());
	for (idx_t i = 0; i < indices.size(); i++) {
		sel.set_index(i, indices[i]);
	}
	chunk->data[0].Dictionary(strings, sel, indices.size());
	chunk->data[1].Dictionary(numbers, sel, indices.size());
	chunk->SetChildCardinality(indices.size());
	return chunk;
}

} // namespace

TEST_CASE("The chunk cache demotes a global dictionary column that arrives as a CONSTANT", "[caching_operator]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &client = *con.context;
	auto &allocator = Allocator::Get(client);

	const string long_string = "a string longer than twelve bytes";
	auto strings = DictionaryVector::CreateReusableGlobalDictionary(LogicalType::VARCHAR, 3);
	strings->data.SetValue(0, Value("x"));
	strings->data.SetValue(1, Value(long_string));
	strings->data.SetValue(2, Value(LogicalType::VARCHAR));
	auto numbers = DictionaryVector::CreateReusableGlobalDictionary(LogicalType::INTEGER, 3);
	for (idx_t i = 0; i < 3; i++) {
		numbers->data.SetValue(i, Value::INTEGER(static_cast<int32_t>(10 * i)));
	}

	// a dictionary chunk seeds the cache, then the strings column arrives as a CONSTANT, then as the dictionary again
	vector<unique_ptr<DataChunk>> chunks;
	chunks.push_back(DictionaryChunk(allocator, strings, numbers, {1, 0, 2, 1, 0}));
	auto constant = DictionaryChunk(allocator, strings, numbers, {1, 1, 1});
	constant->data[0].Reference(Value(long_string), count_t(3));
	chunks.push_back(std::move(constant));
	chunks.push_back(DictionaryChunk(allocator, strings, numbers, {2, 0}));

	PhysicalPlan plan(allocator);
	const vector<LogicalType> types {LogicalType::VARCHAR, LogicalType::INTEGER};
	auto &op = plan.Make<EmitPreparedChunks>(types, chunks).Cast<EmitPreparedChunks>();
	auto &sink = plan.Make<PhysicalOperator>(PhysicalOperatorType::DUMMY_SCAN, types, 0);
	// the cache only engages inside a pipeline that has a sink
	Executor executor(client);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	build_state.SetPipelineSink(pipeline, &sink, 0);
	ThreadContext thread(client);
	ExecutionContext context(client, thread, &pipeline);

	auto gstate = op.GetGlobalOperatorState(client);
	auto state = op.GetOperatorState(context);
	DataChunk input;
	DataChunk output;
	output.Initialize(allocator, types);
	for (idx_t i = 0; i < chunks.size(); i++) {
		output.Reset();
		REQUIRE(op.Execute(context, input, output, *gstate, *state) == OperatorResultType::NEED_MORE_INPUT);
		// every chunk is under the cache threshold, so the cache holds it and emits nothing
		REQUIRE(output.size() == 0);
	}
	output.Reset();
	REQUIRE(op.FinalExecute(context, output, *gstate, *state) == OperatorFinalizeResultType::FINISHED);

	// the strings column was demoted to flat; the numbers column kept its dictionary
	REQUIRE(output.data[0].GetVectorType() == VectorType::FLAT_VECTOR);
	REQUIRE(output.data[1].GetVectorType() == VectorType::DICTIONARY_VECTOR);
	const Value x("x");
	const Value long_value(long_string);
	const Value null_value(LogicalType::VARCHAR);
	const vector<Value> expected_strings {long_value, x,          null_value, long_value, x,
	                                      long_value, long_value, long_value, null_value, x};
	const vector<int32_t> expected_numbers {10, 0, 20, 10, 0, 10, 10, 10, 20, 0};
	REQUIRE(output.size() == expected_strings.size());
	for (idx_t row = 0; row < output.size(); row++) {
		INFO("row " << row);
		REQUIRE(Value::NotDistinctFrom(output.GetValue(0, row), expected_strings[row]));
		REQUIRE(output.GetValue(1, row) == Value::INTEGER(expected_numbers[row]));
	}
}

} // namespace duckdb
