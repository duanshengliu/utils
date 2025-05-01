# -*- coding:utf-8 -*-
"""
This is a tool for analyzing the quantization error of onnx models, including `Graphwise` and `Layerwise` quantization
error analysis.
1. `Graphwise`: Implement by extend additional outputs to the nodes. There will be cumulative quantization error.
2. `Layerwise`: Implement by dividing the quant onnx model into subgraphs of quantized OPs. There will be no cumulative
    quantization error, which can be used to analyze the single operator quantization error.
"""
import os
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper
from collections import OrderedDict
from typing import List, Tuple
from copy import deepcopy

END = '\r'
UNDERLINE = '\033[4m'
RED = '\033[31m'
GREEN = '\033[32m'
CYAN = '\033[36m'
BOLD = '\033[1m'
RESET = '\033[0m'


class MeasurePrinter():
    """Helper class for print top-k record."""

    def __init__(self,
                 collection: List[Tuple[str, float, float, float, float]],
                 outputs_node_name: List[str],
                 quant_model_path: str,
                 top_k: int = None,
                 verbose: bool = True,
                 quant_error_type: str = 'Graphwise',
                 label: str = 'Layer') -> None:
        self.output = []
        self.quant_model_path = quant_model_path
        self.collection = collection
        self.collection_nan = []
        self.collection_without_nan = []
        for data in self.collection:
            if np.isnan(data[1]):
                self.collection_nan.append(data)
            else:
                self.collection_without_nan.append(data)

        self.collection = sorted(
            self.collection_without_nan, key=lambda x: x[1])[::-1]
        largest_element, smallest_element = self.collection[0][1], self.collection[-1][1]
        self.normalized_by = largest_element - smallest_element
        self.min = smallest_element
        max_name_length = len(label)
        self.collection += self.collection_nan
        for i in range(len(self.collection) - 1, -1, -1):
            name, *_ = self.collection[i]
            max_name_length = max(len(name), max_name_length)
            if name in outputs_node_name:
                self.output.append(self.collection.pop(i))
        self.collection = self.output[::-1] + self.collection
        self.max_name_length = max_name_length
        self.quant_error_type = quant_error_type
        self.measure_str = "Mean Absolute Error"
        self.max_blocks = 20
        self.verbose = verbose
        self.top_k = top_k
        self.label = label

    def print_error(self):
        title = (f'{self.label}'
                 f'{" " * (self.max_name_length - len(self.label))} |'
                 f'{" " * 7}{self.measure_str}'
                 f'{" " * (self.max_blocks - len(self.measure_str) + 4)} '
                 f'| Cosine | MaxDiff | MinDiff')
        if not self.verbose:
            return
        if self.collection_nan and ort.__version__ < "1.14.0":
            print(f"\n{RED}[Warning] The onnxruntime version < 1.14.0 may result in some operator's output is None. "
                  f"Please try `pip install onnxruntime==1.14.0`, then try again!{RESET}")
        print(f'{CYAN}The Top{self.top_k} `{self.quant_error_type}` quantization error results are as follows:')
        print(f'{UNDERLINE}{GREEN}' + title)
        csv_path = os.path.abspath(self.quant_model_path).replace(
            '.onnx', f'_{self.quant_error_type}.csv')
        csv_file = open(csv_path, 'w', encoding='utf-8')
        csv_file.write(title.replace('|', ',').replace(' ', '') + '\n')
        # if self.top_k is None, print the quantization error of all operators.
        if self.top_k is None:
            self.top_k = len(self.collection) - len(self.output)
        for i, data in enumerate(self.collection):
            COLOR = f"{RESET}{CYAN}" if i >= len(self.output) else ""
            name, mae, cos_sim, max_diff, min_diff = data
            if np.isnan(mae):
                num_of_blocks = 0
            else:
                normalized_value = (mae - self.min) / \
                    (self.normalized_by + 1e-7)
                num_of_blocks = round(normalized_value * self.max_blocks)
            line = (f'{name}{" " * (self.max_name_length - len(name))} | '
                    f'{"█" * num_of_blocks}{" " * (self.max_blocks - num_of_blocks)}   '
                    f'{mae:7.4f} | '
                    f'{cos_sim:6.4f} | '
                    f'{max_diff:7.4f} | {min_diff:7.4f}')
            # Only print output and top_k quantization error.
            if i < len(self.output) + self.top_k:
                print(f'{COLOR}' + line)
            new_line = line.replace('|', ',').replace(
                '█', ' ').replace(' ', '')
            csv_file.write(new_line + '\n')
        csv_file.close()
        print(
            f'{COLOR}\nSave the complete `{self.quant_error_type}` quantization error CSV file to : {csv_path}{RESET}\n'
        )


def build_output(name):
    """Build output ValueInfoProto List for graph output."""
    output = onnx.ValueInfoProto()
    output.name = name
    output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return [output]


def extend_layer(float_graph, quant_graph, outputs_name, suffix):
    """Extend layer for float graph and quant graph."""
    quant_model_nodes = {node.name for node in quant_graph.node}
    for node in float_graph.node:
        # Ignore constant node.
        if node.op_type == 'Constant':
            continue
        for output in node.output:
            if output in outputs_name:
                continue
            if output + '_DequantizeLinear' in quant_model_nodes:
                float_graph.output.extend(build_output(output))
                quant_graph.output.extend(build_output(output + suffix))


def infer_layer(onnx_model, input_data=None):
    """"Infer the extend layer models and return a dictionary:
    {'layer_name': layer_output}.
    """
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    ort_session = ort.InferenceSession(onnx_model.SerializeToString(),
                                       sess_options=sess_options,
                                       providers=['CPUExecutionProvider'])
    ort_inputs = {}
    for input_ele in ort_session.get_inputs():
        ort_inputs[input_ele.name] = input_data.get(input_ele.name)
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, input_feed=ort_inputs)
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    return ort_outs


def output_name_to_node(onnx_model):
    """Build a mapping between output node names and the current node."""
    output_name_to_node = {}
    for node in onnx_model.graph.node:
        for output_name in node.output:
            output_name_to_node[output_name] = node
    return output_name_to_node


def cosine_similarity(x, y):
    """Calculate cosine similarity."""
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        return np.nan
    return np.dot(x, y) / (x_norm * y_norm)


def layerwise_error_analyse(floatOut, quant_model, inputs, suffix):
    """The quant model will continuously divided into subgraphs
       of quantized OPs."""
    quant_model_copy = deepcopy(quant_model)
    for node in quant_model_copy.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            continue
        for i, node_input in enumerate(node.input):
            new_input_name = f"{node_input}_new_inputs{i}"
            new_node_array = floatOut.get(node_input[:-len(suffix)])
            if new_node_array is None:
                break
            while new_input_name in inputs:
                # avoid `Duplicate definition-site` inputs and
                # they share the same input data: `new_node_array`
                new_input_name += '_'
            new_input = helper.make_tensor_value_info(new_input_name,
                                                      onnx.TensorProto.FLOAT,
                                                      new_node_array.shape)
            node.input[i] = new_input_name
            quant_model_copy.graph.input.extend([new_input])
            inputs[new_input_name] = new_node_array
    layerwise_quantOut = infer_layer(quant_model_copy, inputs)
    return layerwise_quantOut


def cal_quant_error_sum(floatOut, quantOut, collect_map,
                        out_to_node, outputs_name, suffix):
    """Sum up the individual quantization error metrics for each inference."""
    if not collect_map:
        for layer, _ in floatOut.items():
            collect_map[out_to_node[layer].name] = [0, 0, 0, 0]
    for layer, _ in floatOut.items():
        q_layer = layer
        if layer not in outputs_name:
            q_layer += suffix
        if quantOut[q_layer] is None:
            mae = cos_sim = max_diff = min_diff = np.nan
        else:
            abs_diff = np.abs(floatOut[layer] - quantOut[q_layer])
            mae = np.mean(abs_diff)
            cos_sim = cosine_similarity(floatOut[layer], quantOut[q_layer])
            max_diff = np.max(abs_diff)
            min_diff = np.min(abs_diff)
        # node output --> node name
        node = out_to_node[layer]
        key_name = node.name
        n = 1 / len(node.output) if len(node.output) else 1
        collect_map[key_name][0] += n * mae
        collect_map[key_name][1] += n * cos_sim
        collect_map[key_name][2] += n * max_diff
        collect_map[key_name][3] += n * min_diff
    return collect_map


def print_quant_error(quant_error_type, collect_map, data_count, outputs_node_name,
                      quant_model_path, top_k, verbose):
    """Calculate the average of the individual quantization error
        metrics for all inferences and print/display them."""
    collection = [(key_name, *np.array(v_list) / data_count)
                  for key_name, v_list in collect_map.items()]
    MeasurePrinter(collection, outputs_node_name,
                   quant_model_path, top_k, verbose, quant_error_type).print_error()


def progress_bar(iteration, total_iterations, fill='=', bar_length=50):
    """Print the progress status for quantization error analysis."""
    progress = (iteration + 1) / total_iterations * 100
    progress = min(progress, 100)
    filled_length = int(progress / 100 * bar_length)
    bar = f'{fill}' * filled_length + '>' + \
        '.' * (bar_length - filled_length - 1)
    print(f'Progress: [{bar}] {progress:.1f}%', end=END)


def analyse_quantization_error(float_model_path, quant_model_path,
                               input_data=None, top_k=None, verbose=True,
                               suffix="_DequantizeLinear_Output"):
    """Measure the quantization error from a float graph to its quantized graph."""
    # load models
    float_model = onnx.load(float_model_path)
    quant_model = onnx.load(quant_model_path)
    outputs_name = [out_name.name for out_name in float_model.graph.output]
    out_to_node = output_name_to_node(float_model)
    # output --> output node name
    outputs_node_name = [out_to_node[out_name].name for out_name in outputs_name]
    # extend layer for graphwise inference
    extend_layer(float_model.graph, quant_model.graph, outputs_name, suffix)
    if not input_data:
        assert False, "No data provided!"
    data_count = len(input_data)
    graph_collect_map, layer_collect_map = {}, {}
    print("Analyzing quantization accuracy...")
    for i, inputs in enumerate(input_data):
        floatOut = infer_layer(float_model, inputs)
        # graphwise error analyse
        graph_quantOut = infer_layer(quant_model, inputs)

        # layerwise error analyse
        layer_quantOut = layerwise_error_analyse(floatOut, quant_model, inputs, suffix)
        # get graphwise error map
        graph_collect_map = cal_quant_error_sum(floatOut, graph_quantOut, graph_collect_map,
                                                out_to_node, outputs_name, suffix)
        # get layerwise error map
        layer_collect_map = cal_quant_error_sum(floatOut, layer_quantOut, layer_collect_map,
                                                out_to_node, outputs_name, suffix)
        # update progress bar
        progress_bar(i, data_count)
    print()
    print_quant_error('Graphwise', graph_collect_map, data_count, outputs_node_name,
                      quant_model_path, top_k, verbose)
    print(20 * f"{RESET}===")
    print_quant_error('Layerwise', layer_collect_map, data_count, outputs_node_name,
                      quant_model_path, top_k, verbose)
    return graph_collect_map
