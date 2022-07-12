import 'package:neural_tree/src/linalg.dart';
import 'package:neural_tree/src/multi_graph.dart';

import 'component.dart';

class GraphComponentBackwardProducts extends ComponentBackwardProducts {
  MultiGraphBackwardFlow flow;
  GraphComponentBackwardProducts(this.flow);
  FVector get propagatedError {
    return flow.propagatedErrors[0]!;
  }

  Delta get abstractDelta {
    return flow.delta;
  }
}

class GraphComponentForwardProducts extends ComponentForwardProducts {
  MultiGraphForwardFlow flow;
  GraphComponentForwardProducts(this.flow);
  FVector get output => flow.nodeFeeds[flow.graph.outputIndices[0]]!;
}

class GraphComponent extends Component {
  MultiGraph mGraph;
  GraphComponent(this.mGraph)
      : super(mGraph.inputWidth[0], mGraph.outputWidth[0]) {
    if (mGraph.totalInputs != 1 || mGraph.totalOutputs != 1) {
      throw ArgumentError(
          "GraphComponent only takes a graph with single input and single output");
    }
  }
  factory GraphComponent.fromJson(Map<String, dynamic> map) {
    MultiGraph mGraph = MultiGraph.fromJson(map['graph']);
    return GraphComponent(mGraph);
  }

  Map<String, dynamic> toJson() {
    return {"type": "graph", "graph": mGraph.toJson()};
  }

  @override
  Delta zeroDelta() {
    return mGraph.zeroDelta();
  }

  @override
  ComponentBackwardProducts backPropagate(
      ForwardProducts fwdProducts, FVector err) {
    return GraphComponentBackwardProducts(mGraph
        .backPropagateByError((fwdProducts as GraphComponentForwardProducts).flow , [err]));
  }
  

  @override
  FVector feedForward(FVector input) {
    return mGraph.feedForward([input]).single;
  }

  @override
  ComponentForwardProducts produce(FVector input) {
    return GraphComponentForwardProducts(mGraph.produce([input]));
  }

  @override
  void updateWeights(Delta delta) {
    mGraph.update(delta as DeltaList);
  }
}
