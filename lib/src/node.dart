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
      executeChain.add(createImplementation(id,dependencies));
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
      Int32List dependencies = Int32List.fromList(inputNodes.map((input) => assigned[input]!).toList());
      executeChain.add(createImplementation(id,dependencies));
      return id + 1;
    }
    return id;
  }
}
/*
// a 'group' of indices
abstract class MultiInputGroupNode extends Node {
  final List<Node> inputNodes;
  MultiInputGroupNode(this.inputNodes, int width, String name) : super(width, name);
  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {

  }
}
// a 'group' of indices
abstract class SingleInputGroupNode extends Node {
  final List<Node> inputNodes;
  SingleInputGroupNode(this.inputNodes, int width, String name) : super(width, name);
  int assignId(int id, Map<String, int>? inputMapping,
      List<NodeImpl> executeChain, Map<Node, int> assigned) {

  }
}
*/
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



