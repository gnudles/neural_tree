import 'dart:typed_data';

import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';
import 'package:neural_tree/src/node_impl.dart';

/// Node
abstract class Node {
  final String name;
  final int outWidth;
  Node(this.outWidth, this.name);
  NodeImpl createImplementation(int id, Int32List dependencies);
  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned);
}

abstract class SingleInputNode extends Node {
  final Node input;
  SingleInputNode(this.input, int width, String name) : super(width, name);
  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {
    if (!assigned.containsKey(this)) {
      id = input.assignId(id, inputMapping, executeChain, assigned);
      assigned[this] = id;
      Int32List dependencies = Int32List.fromList([assigned[input]!]);
      executeChain.add(createImplementation(id, dependencies));
      return id + 1;
    }
    return id;
  }
}

abstract class MultiInputNode extends Node {
  final List<Node> inputNodes;
  MultiInputNode(this.inputNodes, int width, String name) : super(width, name);
  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {
    if (!assigned.containsKey(this)) {
      inputNodes.forEach((element) {
        id = element.assignId(id, inputMapping, executeChain, assigned);
      });
      assigned[this] = id;
      Int32List dependencies = Int32List.fromList(
          inputNodes.map((input) => assigned[input]!).toList());
      executeChain.add(createImplementation(id, dependencies));
      return id + 1;
    }
    return id;
  }
}

abstract class GroupNode extends Node {
  int outputsCount;
  GroupNode(this.outputsCount, int width, String name) : super(width, name);
}

// a 'group' of indices
abstract class MultiInputGroupNode extends GroupNode {
  final GroupNode input;

  MultiInputGroupNode(this.input, super.outputsCount, super.width, super.name);

  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {
    if (!assigned.containsKey(this)) {
      id = input.assignId(id, inputMapping, executeChain, assigned);
      assigned[this] = id;
      Int32List dependencies = Int32List.fromList(List.generate(input.outputsCount, (index) => assigned[input]!+index));
      executeChain.add(createImplementation(id, dependencies));
      return id + outputsCount;
    }
    return id;
  }
}

// a 'group' of indices
abstract class SingleInputGroupNode extends GroupNode {
  final Node inputNode;
  SingleInputGroupNode(this.inputNode, super.outputsCount, super.width, super.name);
  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {
    if (!assigned.containsKey(this)) {
      id = inputNode.assignId(id, inputMapping, executeChain, assigned);
      assigned[this] = id;
      Int32List dependencies = Int32List.fromList([assigned[inputNode]!]);
      executeChain.add(createImplementation(id, dependencies));
      return id + outputsCount;
    }
    return id;
  }
}

class InputNode extends Node {
  InputNode(int width, String name) : super(width, name);

  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {
    if (!assigned.containsKey(this)) {
      assigned[this] = id;
      if (inputMapping != null) inputMapping[name] = id;
      return id + 1;
    }
    return id;
  }

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    // This is Ok, we never call this
    throw UnimplementedError();
  }
}
