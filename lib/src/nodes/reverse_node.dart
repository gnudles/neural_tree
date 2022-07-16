import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class ReverseNode extends SingleInputNode {
  ReverseNode(Node input, {String name = ""})
      : super(input, input.outWidth, name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return ReverseNodeImpl(id, name, outWidth, dependencies);
  }
}

class ReverseNodeImpl extends NodeImpl {
  ReverseNodeImpl(super.id, super.name, super.outWidth, super.dependencies);
  factory ReverseNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    return ReverseNodeImpl(id, name, outWidth, dependencies);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type': 'reverse',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    feeds[id] = feeds[dependencies.single]!.reversed();
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var reversedErrors = propagatedErrors[id]!.reversed();
    int depId = dependencies.single;
    if (propagatedErrors[depId] == null) {
      propagatedErrors[depId] = reversedErrors;
    } else {
      propagatedErrors[depId]!.add(reversedErrors);
    }
  }

  @override
  void update(Delta? delta, double maxWeight , double maxBias ) {}
}
