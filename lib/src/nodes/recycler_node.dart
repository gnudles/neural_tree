import 'dart:typed_data';

import '../component.dart';
import '../linalg.dart';
import '../node.dart';
import '../node_impl.dart';


class RecyclerForwardProducts extends ForwardProducts {
  List<ComponentForwardProducts> products;
  RecyclerForwardProducts(this.products);
}

class RecyclerBackwardProducts extends BackwardProducts {
  List<ComponentBackwardProducts> products;
  RecyclerBackwardProducts(this.products);
}

/// Convolution layer
class RecyclerNode extends MultiInputNode {
  final Component recycled;
  RecyclerNode(this.recycled, List<Node> inputNodes,
      { String name = ""})
      : super(
            inputNodes,
                 recycled.outWidth * inputNodes.length,
            name);

  @override
  NodeImpl createImplementation(int id, Int32List dependencies) {
    return RecyclerNodeImpl(id, name, outWidth, dependencies, recycled);
  }
}

/// Convolution layer
class RecyclerNodeImpl extends NodeImpl {
  final Component recycled;
  RecyclerNodeImpl(
      super.id, super.name, super.outWidth, super.dependencies, this.recycled);
  factory RecyclerNodeImpl.fromJson(Map<String, dynamic> map) {
    int id = map['id'];
    int outWidth = map['outWidth'];
    Int32List dependencies =
        Int32List.fromList((map['dependencies'] as List<dynamic>).cast());
    String name = map['name'];
    Component recycled = componentLoaders[map['recycled']['type']]!(map['recycled']);

    return RecyclerNodeImpl(
        id, name, outWidth, dependencies, recycled);
  }
  @override
  Map<String, dynamic> toJson() {
    return {
      'node_type':'recycler',
      'id': id,
      'name': name,
      'outWidth': outWidth,
      'dependencies': dependencies,
      'recycled': recycled.toJson(),
    };
  }

  @override
  void backPropagate(List<BackwardProducts> backProducts, List<Delta?> deltas,
      List<FVector?> propagatedErrors, List<ForwardProducts> fwdProducts) {
    var fwdList = fwdProducts[id] as RecyclerForwardProducts;
    var propErr = propagatedErrors[id]!;
    List<ComponentBackwardProducts> bProductsList =
        List.filled(dependencies.length, ComponentBackwardProducts());

    var delta = recycled.zeroDelta();
    for (int i = 0; i < dependencies.length; ++i) {
      var bProd = recycled.backPropagate(fwdList.products[i],
          propErr.slice(recycled.outWidth * i, recycled.outWidth));
      bProductsList[i] = bProd;
      // place propagated errors
      if (propagatedErrors[dependencies[i]] == null) {
        propagatedErrors[dependencies[i]] = bProd.propagatedError.clone();
      } else {
        propagatedErrors[dependencies[i]]!.add(bProd.propagatedError);
      }
      //add up the delta
      delta.add(bProd.abstractDelta);
    }
    delta.scale(1 / dependencies.length);

    backProducts[id] = RecyclerBackwardProducts(bProductsList);

    deltas[id] = delta;
  }

  @override
  void execute(List<FVector?> feeds, List<ForwardProducts>? fwdProducts) {
    FVector result;
    if (fwdProducts == null) {
      var outputsList = dependencies.map((dep) {
        FVector input = feeds[dep]!;
        return recycled.feedForward(input);
      }).toList();
      result = FVector.join(outputsList);
    } else {
      List<FVector?> outputsList = List.filled(dependencies.length, null);
      List<ComponentForwardProducts?> prodList =
          List.filled(dependencies.length, null);
      for (int i = 0; i < dependencies.length; ++i) {
        FVector input = feeds[dependencies[i]]!;
        prodList[i] = recycled.produce(input);
        outputsList[i] = prodList[i]!.output;
      }

      fwdProducts[id] = RecyclerForwardProducts(prodList.cast());
      result = FVector.join(outputsList.cast());
    }
    feeds[id] = result;
  }

  @override
  void update(Delta? delta) {
    recycled.updateWeights(delta!);
  }

  Delta? zeroDelta() => recycled.zeroDelta();
}
