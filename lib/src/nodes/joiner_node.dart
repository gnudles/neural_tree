import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class JoinerNode extends MultiInputNode {
  JoinerNode(List<Node> inputNodes, {String name = ""})
      : super(
            inputNodes,
            inputNodes.fold<int>(0,
                (previousValue, element) => previousValue + element.outWidth),
            name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    Int32List lengths = Int32List.fromList(
        inputNodes.map((n) => n.outWidth).toList(growable: false));
    int outWidth = inputNodes.fold<int>(
        0, (previousValue, element) => previousValue + element.outWidth);
        
    return JoinerNodeImpl(id, name, outWidth, dependencies, lengths);
  }
}

class JoinerNodeImpl extends NodeImpl {
  Int32List lengths;
  JoinerNodeImpl(super.id, super.name, super.outWidth, super.dependencies,
      this.lengths);

  factory JoinerNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    Int32List lengths = Int32List.fromList((map['lengths'] as List<dynamic>).cast());
    return JoinerNodeImpl(
        id, name, outWidth, dependencies, lengths);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type':'joiner',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
      'lengths': lengths,
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    feeds[id] = FVector.join(
        dependencies.map((e) => feeds[e]!).toList(growable: false));
  }

  @override
  void backPropagate(
          List<BackwardProducts> backProducts,
          List<Delta?> deltas,
      List<FVector?> propagatedErrors,
      List<ForwardProducts> fwdProducts) {
    var joinedErrors = propagatedErrors[id]!;
    List<FVector> sliced = joinedErrors.sliceByLengths(lengths, 0);
    for (int i = 0; i < dependencies.length; ++i) {
      int depId = dependencies[i];
      if (propagatedErrors[depId] == null) {
        propagatedErrors[depId] = sliced[i];
      } else {
        propagatedErrors[depId]!.add(sliced[i]);
      }
    }
  }

  @override
  void update(Delta? delta) {}
}
