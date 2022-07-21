import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class SumNode extends MultiInputNode {
  SumNode(List<Node> inputNodes, {String name = ""})
      : super(inputNodes, inputNodes[0].outWidth, name) {
    if (!inputNodes.skip(1).every((element) => element.outWidth == outWidth)) {
      throw ArgumentError("sizes of inputs are not equal to each other.");
    }
  }

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return SumNodeImpl(id, name, outWidth, dependencies);
  }
}

class SumNodeImpl extends NodeImpl {
  SumNodeImpl(super.id, super.name, super.outWidth, super.dependencie);

  factory SumNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    return SumNodeImpl(id, name, outWidth, dependencies);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type': 'sum',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    var sum = FVector.zero(outWidth);
    dependencies.forEach((dep) {
      sum.add(feeds[dep]!);
    });
    feeds[id] = sum;
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var propErr = propagatedErrors[id]!;
    for (int i = 0; i < dependencies.length; ++i) {
      int depId = dependencies[i];
      if (propagatedErrors[depId] == null) {
        propagatedErrors[depId] = propErr.clone();
      } else {
        propagatedErrors[depId]!.add(propErr);
      }
    }
  }

  @override
  void update(Delta? delta, double maxWeight, double maxBias) {}
}
