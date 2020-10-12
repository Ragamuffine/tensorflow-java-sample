package cafe.aulait;

import org.tensorflow.*;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Gradients;
import java.util.Arrays;

public class GradientsTest {
    public static void test1() {
	Graph graph = new Graph();
	// 2
	Operation two = graph.opBuilder( "Const", "a" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .setAttr( "value", Tensor.create( 2.0, Double.class ) )
	    .build();
	// x
	Operation x = graph.opBuilder( "Placeholder", "x" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .build();
	// y = x^2
	Operation y = graph.opBuilder( "Pow", "y" )
	    .addInput (x.output (0))
	    .addInput (two.output (0))
	    .build();
	Session session = new Session (graph);
	Tensor tensor1 = session.runner()
	    .fetch ("y")
	    .feed( "x", Tensor.create( 5.0, Double.class ) )
	    .run()
	    .get (0).expect (Double.class);
	System.out.println (tensor1.doubleValue());
	Scope scope = new Scope (graph);
	Gradients gradients = Gradients.create( scope, y.output (0), Arrays.asList (x.output (0)) );
	Tensor tensor2 = session.runner()
	    .feed( "x", Tensor.create( 5.0, Double.class ) )
	    .fetch (gradients.dy (0))
	    .run()
	    .get (0).expect (Double.class);
	System.out.println (tensor2.doubleValue());
    }

    public static void test2() {
	Graph graph = new Graph();
	// x
	Operation x = graph.opBuilder( "Placeholder", "x" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .build();
	// y = log x
	Operation y = graph.opBuilder( "Log", "y" )
	    .addInput (x.output (0))
	    .build();
	Session session = new Session (graph);
	Tensor tensor1 = session.runner()
	    .fetch ("y")
	    .feed( "x", Tensor.create( 5.0, Double.class ) )
	    .run()
	    .get (0).expect (Double.class);
	System.out.println (tensor1.doubleValue());
	Scope scope = new Scope (graph);
	Gradients gradients = Gradients.create( scope, y.output (0), Arrays.asList (x.output (0)) );
	Tensor tensor2 = session.runner()
	    .feed( "x", Tensor.create( 5.0, Double.class ) )
	    .fetch (gradients.dy (0))
	    .run()
	    .get (0).expect (Double.class);
	System.out.println (tensor2.doubleValue());
    }

    public static void test3() {
	Graph graph = new Graph();
	// x
	Operation x = graph.opBuilder( "Placeholder", "x" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .build();
	// y = exp x
	Operation y = graph.opBuilder( "Exp", "y" )
	    .addInput (x.output (0))
	    .build();
	Session session = new Session (graph);
	Tensor tensor1 = session.runner()
	    .fetch ("y")
	    .feed( "x", Tensor.create( 5.0, Double.class ) )
	    .run()
	    .get (0).expect (Double.class);
	System.out.println (tensor1.doubleValue());
	Scope scope = new Scope (graph);
	Gradients gradients = Gradients.create( scope, y.output (0), Arrays.asList (x.output (0)) );
	Tensor tensor2 = session.runner()
	    .feed( "x", Tensor.create( 5.0, Double.class ) )
	    .fetch (gradients.dy (0))
	    .run()
	    .get (0).expect (Double.class);
	System.out.println (tensor2.doubleValue());
    }

    public static void main( String args[] ) {
	try {
	    test1();
	    test2();
	    test3();
	}
	catch ( Exception e ) {
	    e.printStackTrace();
	}
    }
}
