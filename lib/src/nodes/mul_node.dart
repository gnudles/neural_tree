import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class MulNode extends MultiInputNode {
  MulNode(List<Node> inputNodes, {String name = ""})
      : super(inputNodes, inputNodes[0].outWidth, name) {
    if (!inputNodes.skip(1).every((element) => element.outWidth == outWidth)) {
      throw ArgumentError("sizes of inputs are not equal to each other.");
    }
  }

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return MulNodeImpl(id, name, outWidth, dependencies);
  }
}

class MultiplicationForwardProducts extends ComponentForwardProducts {
  List<FVector> sources;
  FVector product;
  MultiplicationForwardProducts(this.sources, this.product);
  FVector get output => product;
}

class MulNodeImpl extends NodeImpl {
  MulNodeImpl(super.id, super.name, super.outWidth, super.dependencie);

  factory MulNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    return MulNodeImpl(id, name, outWidth, dependencies);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type': 'mul',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    var product = feeds[dependencies[0]]!.clone();
    dependencies.skip(1).forEach((dep) {
      product.dotProduct(feeds[dep]!);
    });
    feeds[id] = product;
    if (fwdProducts != null) {
      fwdProducts[id] = MultiplicationForwardProducts(
          dependencies.map((dep) => feeds[dep]!).toList(growable: false),
          product);
    }
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var propErr = propagatedErrors[id]!;
    var fwd = fwdProducts[id] as MultiplicationForwardProducts;
    for (int i = 0; i < dependencies.length; ++i) {
      int depId = dependencies[i];
      FVector allButMe = propErr.clone();

      //TODO: better algorithm for 'all but me' multiplication.
      for (int j = 0; j < dependencies.length; ++j) {
        if (j != i) {
          allButMe.dotProduct(fwd.sources[dependencies[j]]);
        }
      }
      
      if (propagatedErrors[depId] == null) {
        propagatedErrors[depId] = allButMe;
      } else {
        propagatedErrors[depId]!.add(allButMe);
      }
    }
  }

  @override
  void update(Delta? delta, double maxWeight, double maxBias) {}
}
