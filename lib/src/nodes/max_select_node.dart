import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class MaxSelectNode extends MultiInputNode {
  MaxSelectNode(List<Node> inputNodes, {String name = ""})
      : super(inputNodes, inputNodes[0].outWidth, name) {
    assert(inputNodes.isNotEmpty);
    assert(inputNodes
        .skip(1)
        .every((element) => element.outWidth == inputNodes[0].outWidth));
  }

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return MaxSelectNodeImpl(id, name, outWidth, dependencies);
  }
}

class MaxSelectForwardProducts extends ComponentForwardProducts {
  final FVector max;
  final Int32List selectors;
  const MaxSelectForwardProducts(this.max, this.selectors);
  FVector get output => max;
}

class MaxSelectNodeImpl extends NodeImpl {
  MaxSelectNodeImpl(super.id, super.name, super.outWidth, super.dependencies);
  factory MaxSelectNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    return MaxSelectNodeImpl(id, name, outWidth, dependencies);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type': 'max_select',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    FVector maxVector = FVector.zero(outWidth);
    Int32List selectors = Int32List(outWidth);
    for (int i = 0; i < outWidth; ++i) {
      double maxVal = double.negativeInfinity;
      int maxDep = -1;
      for (int d in dependencies) {
        if (feeds[d]![i] > maxVal) {
          maxVal = feeds[d]![i];
          maxDep = d;
        }
      }
      maxVector[i] = maxVal;
      selectors[i] = maxDep;
    }
    feeds[id] = maxVector;
    if (fwdProducts != null) {
      fwdProducts[id] = MaxSelectForwardProducts(maxVector, selectors);
    }
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var maxErrors = propagatedErrors[id]!;
    var fwd = fwdProducts[id] as MaxSelectForwardProducts;
    for (int i = 0; i < outWidth; ++i) {
      int d = fwd.selectors[i];
      if (propagatedErrors[d] == null) {
        propagatedErrors[d] = FVector.zero(outWidth);
      }
      propagatedErrors[d]![i] += maxErrors[i];
    }
  }

  @override
  void update(Delta? delta) {}
}
