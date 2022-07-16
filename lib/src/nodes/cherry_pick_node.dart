import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class CherryPickNode extends SingleInputNode {
  final List<int> indices;

  CherryPickNode(this.indices, Node input, {String name = ""})
      : super(input, indices.length, name);
  CherryPickNode.random(int length, Node input, {String name = ""})
      : indices = List.generate(input.outWidth, (i) => i)
          ..shuffle()
          ..removeRange(length, input.outWidth),
        super(input, length, name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return CherryPickNodeImpl(id, name, outWidth, dependencies,
        Int32List.fromList(indices), input.outWidth);
  }
}

class CherryPickNodeImpl extends NodeImpl {
  final Int32List indices;
  final int inWidth;
  CherryPickNodeImpl(super.id, super.name, super.outWidth, super.dependencies,
      this.indices, this.inWidth);
  factory CherryPickNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    Int32List indices = Int32List.fromList((map['indices'] as List<dynamic>).cast());
    int inWidth = map['inWidth'];
    return CherryPickNodeImpl(
        id, name, outWidth, dependencies, indices, inWidth);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type':'cherry',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
      'indices': indices,
      'inWidth': inWidth,
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    feeds[id] = feeds[dependencies.single]!.cherryPick(indices);
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var cherryPickErrors = propagatedErrors[id]!;
    var wide = FVector.zero(inWidth);
    for (int i = 0; i < indices.length; ++i) {
      wide[indices[i]] = cherryPickErrors[i];
    }
    int depId = dependencies.single;
    if (propagatedErrors[depId] == null) {
      propagatedErrors[depId] = wide;
    } else {
      propagatedErrors[depId]!.add(wide);
    }
  }

  @override
  void update(Delta? delta, double maxWeight , double maxBias )  {}
}
