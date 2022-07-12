import 'dart:typed_data';

import 'package:neural_tree/src/biclique.dart';
import 'package:neural_tree/src/biregular.dart';
import 'package:neural_tree/src/graph_component.dart';
import 'package:neural_tree/src/uniform.dart';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';

class ComponentNode extends SingleInputNode {
  Component component;
  ComponentNode(this.component, Node input, {String name = ""})
      : super(input, component.outWidth, name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return ComponentNodeImpl(id, name, dependencies, component);
  }
}



class ComponentNodeImpl extends NodeImpl {
  Component component;
  ComponentNodeImpl(int id, String name, Int32List dependencies, this.component)
      : super(id, name, component.outWidth, dependencies);

    factory ComponentNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    //int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    Component component = componentLoaders[map['component']['type']]!(map['component']);

    return ComponentNodeImpl(
        id, name, dependencies, component);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type':'component',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
      'component': component.toJson(),
    };
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    FVector input = feeds[dependencies[0]]!;
    FVector result;
    if (fwdProducts == null) {
      result = component.feedForward(input);
    } else {
      var products = component.produce(input);
      fwdProducts[id] = products;
      result = products.output;
    }
    feeds[id] = result;
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var bproducts =
        component.backPropagate(fwdProducts[id], propagatedErrors[id]!);
    backProducts[id] = bproducts;
    var depId = dependencies.single;
    if (propagatedErrors[depId] == null) {
      propagatedErrors[depId] = bproducts.propagatedError.clone();
    } else {
      propagatedErrors[depId]!.add(bproducts.propagatedError);
    }
    deltas[id] = bproducts.abstractDelta;
  }
  Delta? zeroDelta() => component.zeroDelta();

  @override
  void update(Delta? delta) {
    component.updateWeights(delta!);
  }
}
