import 'package:neural_tree/src/kapandaria_extensions.dart';
import 'package:neural_tree/src/linalg.dart';
import 'package:neural_tree/src/node.dart';
import 'package:neural_tree/src/node_impl.dart';
import 'package:neural_tree/src/nodes/cherry_pick_node.dart';
import 'package:neural_tree/src/nodes/component_node.dart';
import 'package:neural_tree/src/nodes/joiner_node.dart';
import 'package:neural_tree/src/nodes/max_select_node.dart';
import 'package:neural_tree/src/nodes/recycler_node.dart';
import 'package:neural_tree/src/nodes/reverse_node.dart';
import 'package:neural_tree/src/nodes/range_node.dart';
import 'package:neural_tree/src/nodes/sum_node.dart';
import 'component.dart';
import 'nodes/activation_node.dart';
import 'nodes/mul_node.dart';

List<NodeImpl> sortChain(
    List<NodeImpl> tempChain, List<int> availableDependencies) {
  List<NodeImpl> executeChain = [];
  while (tempChain.isNotEmpty) {
    var toBeAdded = tempChain
        .indicesWhere((node) => node.dependencies
            .every((dep) => availableDependencies.contains(dep)))
        .toList();
    if (toBeAdded.isEmpty) throw ArgumentError("dependency loop, or bug");

    toBeAdded.reversed.forEach((index) {
      var node = tempChain[index];
      executeChain.add(node);
      availableDependencies.add(node.id);
      tempChain.removeAt(index);
    });
  }
  return executeChain;
}

Map<String, NodeImpl Function(Map<String, dynamic>)> nodeTypeLoaders = {
  'cherry': (map) => CherryPickNodeImpl.fromJson(map),
  'component': (map) => ComponentNodeImpl.fromJson(map),
  'recycler': (map) => RecyclerNodeImpl.fromJson(map),
  'joiner': (map) => JoinerNodeImpl.fromJson(map),
  'max_select': (map) => MaxSelectNodeImpl.fromJson(map),
  'range': (map) => RangeNodeImpl.fromJson(map),
  'reverse': (map) => ReverseNodeImpl.fromJson(map),
  'sum': (map) => SumNodeImpl.fromJson(map),
  'mul': (map) => MulNodeImpl.fromJson(map),
  'activation': (map) => ActivationNodeImpl.fromJson(map),
};

/// Graph with multiple inputs and multiple outputs
class MultiGraph {
  int totalInputs;
  int totalOutputs;

  final List<NodeImpl> executeChain;
  final List<int> inputWidth;
  final List<String> inputNames;
  final List<int> outputIndices;
  final List<int> outputWidth;
  final List<String> outputNames;

  // Total node count (including the virtual input nodes)
  int totalNodes;

  MultiGraph._(
      this.totalInputs,
      this.totalOutputs,
      this.executeChain,
      this.inputWidth,
      this.inputNames,
      this.outputIndices,
      this.outputWidth,
      this.outputNames,
      this.totalNodes);
  factory MultiGraph.fromJson(Map<String, dynamic> map) {
    int totalInputs = map["totalInputs"];
    int totalOutputs = map["totalOutputs"];

    List<NodeImpl> executeChain = (map["executeChain"] as List<dynamic>)
        .cast<Map<String, dynamic>>()
        .map((jnode) {
      String nodeType = jnode["node_type"];
      if (!nodeTypeLoaders.containsKey(nodeType)) {
        throw ArgumentError("invalid node type: $nodeType");
      }
      return nodeTypeLoaders[nodeType]!(jnode);
    }).toList();
    List<int> inputWidth = (map["inputWidth"] as List<dynamic>).cast();
    List<String> inputNames = (map["inputNames"] as List<dynamic>).cast();
    List<int> outputIndices = (map["outputIndices"] as List<dynamic>).cast();
    List<int> outputWidth = (map["outputWidth"] as List<dynamic>).cast();
    List<String> outputNames = (map["outputNames"] as List<dynamic>).cast();
    int totalNodes = map["totalNodes"];
    return MultiGraph._(totalInputs, totalOutputs, executeChain, inputWidth,
        inputNames, outputIndices, outputWidth, outputNames, totalNodes);
  }
  factory MultiGraph.fromOutputNodes(
      List<Node> outputNodes, List<InputNode> inputNodes) {
    int id = 0;
    var temporaryExecuteChain = <NodeImpl>[];
    var assigned = <Node, int>{};
    Map<String, int> inputMapping = {};
    inputNodes.forEach((node) {
      id = node.assignId(id, inputMapping, temporaryExecuteChain, assigned);
    });
    outputNodes.forEach((node) {
      id = node.assignId(id, inputMapping, temporaryExecuteChain, assigned);
    });
    if (inputNodes.length != inputMapping.length) {
      throw ArgumentError(
          "missing or too many input nodes given by inputNodes");
    }
    List<String> inputNames = inputNodes.map((iNode) => iNode.name).toList();
    List<int> inputWidth = inputNodes.map((iNode) => iNode.outWidth).toList();

    List<int> outputIndices =
        outputNodes.map((oNode) => assigned[oNode]!).toList();
    List<int> outputWidth = outputNodes.map((oNode) => oNode.outWidth).toList();

    List<String> outputNames = outputNodes.map((oNode) => oNode.name).toList();
    int totalNodes;
    totalNodes = id;
    List<NodeImpl> executeChain;
    executeChain = sortChain(temporaryExecuteChain, [...inputMapping.values]);
    int totalInputs = inputMapping.length;
    int totalOutputs = outputIndices.length;
    return MultiGraph._(totalInputs, totalOutputs, executeChain, inputWidth,
        inputNames, outputIndices, outputWidth, outputNames, totalNodes);
  }
  List<FVector> feedForward(Iterable<FVector> inputs) {
    if (inputs.length != totalInputs) {
      throw ArgumentError("number of inputs does not match");
    }
    List<FVector?> nodeFeeds = List.filled(totalNodes, null);
    var inputIter = inputs.iterator;
    for (int i = 0; i < totalInputs; ++i) {
      inputIter.moveNext();
      nodeFeeds[i] = inputIter.current;
    }
    executeChain.forEach((element) {
      element.execute(nodeFeeds, null);
    });
    return outputIndices
        .map((index) => nodeFeeds[index]!)
        .toList(growable: false);
  }

  MultiGraphForwardFlow produce(Iterable<FVector> inputs) {
    if (inputs.length != totalInputs) {
      throw ArgumentError("number of inputs does not match");
    }
    List<FVector?> nodeFeeds = List.filled(totalNodes, null);
    List<ForwardProducts> fwdProducts =
        List.filled(totalNodes, ForwardProducts());
    var inputIter = inputs.iterator;
    for (int i = 0; i < totalInputs; ++i) {
      inputIter.moveNext();
      nodeFeeds[i] = inputIter.current;
    }
    executeChain.forEach((element) {
      element.execute(nodeFeeds, fwdProducts);
    });
    return MultiGraphForwardFlow(this, nodeFeeds, fwdProducts);
  }

  MultiGraphBackwardFlow backPropagateByTarget(
      MultiGraphForwardFlow forwardFlow, Iterable<FVector> target,
      {double maxErrClipAbove = 0.5}) {
    assert(forwardFlow.graph == this);
    var targetIter = target.iterator;

    List<FVector> errors = List.generate(totalOutputs, (index) {
      if (!targetIter.moveNext()) {
        throw ArgumentError("missing more targets");
      }
      return forwardFlow.nodeFeeds[outputIndices[index]]! - targetIter.current;
    });
    return backPropagateByError(forwardFlow, errors,
        maxErrClipAbove: maxErrClipAbove);

    /*
    netOutput - target;
    */
  }

  MultiGraphBackwardFlow backPropagateByError(
      MultiGraphForwardFlow forwardFlow, Iterable<FVector> errors,
      {double maxErrClipAbove = 0.5}) {
    if (errors.length != totalOutputs)
      throw ArgumentError("errors length is incorrect");
    List<FVector?> propagatedErrors = List.filled(totalNodes, null);
    List<Delta?> deltas = List.filled(totalNodes, null);
    List<BackwardProducts> backProducts =
        List.filled(totalNodes, BackwardProducts());
    var errorsIter = errors.iterator;
    for (int i = 0; i < totalOutputs; ++i) {
      errorsIter.moveNext();
      propagatedErrors[outputIndices[i]] = errorsIter.current;
    }
    executeChain.reversed.forEach((node) {
      //clip the error, so we don't get Nan everywhere.
      if (maxErrClipAbove != 0.0) {
        propagatedErrors[node.id]!.clamp(-maxErrClipAbove, maxErrClipAbove);
      }
      //do the actual stuff.
      node.backPropagate(
          backProducts, deltas, propagatedErrors, forwardFlow.fwdProducts);
    });
    return MultiGraphBackwardFlow(
        propagatedErrors, backProducts, DeltaList(deltas));
  }

  DeltaList zeroDelta() {
    List<Delta?> deltas = List.filled(totalNodes, null);
    executeChain.forEach((node) {
      deltas[node.id] = node.zeroDelta();
    });
    return DeltaList(deltas);
  }

  void update(DeltaList delta, [double maxWeight = 0.2, double maxBias = 0.2]) {
    executeChain.forEach((node) {
      node.update(delta.deltas[node.id], maxWeight, maxBias);
    });
  }

  Map<String, dynamic> toJson() {
    return {
      "totalInputs": totalInputs,
      "totalOutputs": totalOutputs,
      "executeChain": executeChain.map((e) => e.toJson()).toList(),
      "inputWidth": inputWidth,
      "inputNames": inputNames,
      "outputIndices": outputIndices,
      "outputWidth": outputWidth,
      "outputNames": outputNames,
      "totalNodes": totalNodes,
    };
  }
}

class MultiGraphForwardFlow {
  MultiGraph graph;
  List<FVector?> nodeFeeds;
  List<ForwardProducts> fwdProducts;
  MultiGraphForwardFlow(this.graph, this.nodeFeeds, this.fwdProducts);
}

class MultiGraphBackwardFlow {
  List<FVector?> propagatedErrors;
  List<BackwardProducts> backProducts;
  DeltaList delta;
  MultiGraphBackwardFlow(this.propagatedErrors, this.backProducts, this.delta);
}

class DeltaList extends Delta {
  List<Delta?> deltas;
  DeltaList(this.deltas);
  factory DeltaList.fromJson(Map<String, dynamic> map) {
    List<dynamic> deltas = map['d'];
    return DeltaList(deltas.map((e) {
      if (e is Map<String, dynamic>) {
        return Delta.fromJson(e);
      }
      return null;
    }).toList());
  }
  @override
  void add(Delta other) {
    if (other is DeltaList && other.deltas.length == this.deltas.length) {
      for (int i = 0; i < deltas.length; ++i) {
        this.deltas[i]?.add(other.deltas[i]!);
      }
    }
  }

  @override
  void scale(double factor) {
    for (int i = 0; i < deltas.length; ++i) {
      this.deltas[i]?.scale(factor);
    }
  }

  @override
  void clamp(double maxVal) {
    for (int i = 0; i < deltas.length; ++i) {
      this.deltas[i]?.clamp(maxVal);
    }
  }

  @override
  double minAbsDelta() {
    double minAbsDelta = 0;
    double currentMin;
    for (int i = 1; i < deltas.length; ++i) {
      var currentDelta = deltas[i];
      if (currentDelta != null) {
        currentMin = currentDelta.minAbsDelta();
        if (minAbsDelta == 0 || currentMin < minAbsDelta) {
          minAbsDelta = currentMin;
        }
      }
    }
    return minAbsDelta;
  }

  @override
  Map<String, dynamic> toJson() {
    return {
      'type': 'graph',
      'd': deltas.map((e) {
        if (e == null) {
          return {};
        }
        return e.toJson();
      }).toList()
    };
  }
}
