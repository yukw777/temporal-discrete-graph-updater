"""
Generate a smaller test dataset with the specified number of examples
from the given dataset.
"""
import json
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("data_file")
parser.add_argument("test_data_file")
parser.add_argument("num_examples", type=int)
args = parser.parse_args()

with open(args.data_file, "r") as f:
    data = json.load(f)

graph_index = json.loads(data["graph_index"])
examples = data["examples"][: args.num_examples]

# extract graphs
prev_seen_graph_idx = set(str(e["previous_graph_seen"]) for e in examples)
graphs = {g_id: graph_index["graphs"][g_id] for g_id in prev_seen_graph_idx}

# extract relations
relation_idx = set(str(r_id) for r_id in itertools.chain.from_iterable(graphs.values()))
relations = {r_id: graph_index["relations"][r_id] for r_id in relation_idx}

sub_graph_index = {
    "graphs": graphs,
    "entities": graph_index["entities"],
    "relations": relations,
    "relation_types": graph_index["relation_types"],
}

test_data = {"graph_index": json.dumps(sub_graph_index), "examples": examples}

with open(args.test_data_file, "w") as f:
    json.dump(test_data, f)
