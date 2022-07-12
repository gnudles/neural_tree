import 'dart:typed_data';

import 'package:neural_tree/src/component.dart';
import 'package:neural_tree/src/linalg.dart';

abstract class Trainable {}

abstract class NodeImpl {
  /// The index of current node in node vector
  final int id;

  /// The width of output vector. Each node has a fixed output width
  final int outWidth;

  /// The indices of nodes we depend on.
  final Int32List dependencies;

  /// Name of the node.
  final String name;

  NodeImpl(this.id, this.name, this.outWidth, this.dependencies);
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts);
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts);
  Delta? zeroDelta() => null;
  void update(Delta? delta);
  Map<String, dynamic> toJson();
}
