import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class RangeNode extends SingleInputNode {
  final int from;
  RangeNode(this.from, int length, Node input, {String name = ""})
      : super(input, length, name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return RangeNodeImpl(
        id, name, outWidth, dependencies, from, input.outWidth);
  }
}

class RangeNodeImpl extends NodeImpl {
  final int from;
  final int inWidth;
  RangeNodeImpl(super.id, super.name, super.outWidth, super.dependencies,
      this.from, this.inWidth);

  factory RangeNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    int from = map['from'];
    int inWidth = map['inWidth'];
    return RangeNodeImpl(
        id, name, outWidth, dependencies, from, inWidth);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type':'range',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
      'from': from,
      'inWidth': inWidth,
    };
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts,
  List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var rangedErrors = propagatedErrors[id]!;
    var wide = FVector.zero(inWidth);
    for (int i = 0; i < outWidth; ++i) {
      wide[from + i] = rangedErrors[i];
    }
    int depId = dependencies.single;
    if (propagatedErrors[depId] == null) {
      propagatedErrors[depId] = wide;
    } else {
      propagatedErrors[depId]!.add(wide);
    }
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    feeds[id] = feeds[dependencies.single]!.slice(from, outWidth);
  }

  @override
  void update(Delta? delta) {
    // TODO: implement update
  }
}
