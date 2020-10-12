package cafe.aulait;

// https://www.baeldung.com/tensorflow-java

import org.tensorflow.*;

public class Basic {
    public static void sample() {
	Graph graph = new Graph();
	// a = 3.0
	Operation a = graph.opBuilder( "Const", "a" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .setAttr( "value", Tensor.create( 3.0, Double.class ) )
	    .build();
	// b = 2.0
	Operation b = graph.opBuilder( "Const", "b" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .setAttr( "value", Tensor.create( 2.0, Double.class ) )
	    .build();
	// x
	Operation x = graph.opBuilder( "Placeholder", "x" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .build();
	// y
	Operation y = graph.opBuilder( "Placeholder", "y" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .build();
	// ax
	Operation ax = graph.opBuilder( "Mul", "ax" )
	    .addInput (a.output (0))
	    .addInput (x.output (0))
	    .build();
	// by
	Operation by = graph.opBuilder( "Mul", "by" )
	    .addInput (b.output (0))
	    .addInput (y.output (0))
	    .build();
	// z = ax + by
	Operation z = graph.opBuilder( "Add", "z" )
	    .addInput (ax.output (0))
	    .addInput (by.output (0))
	    .build();
	Session session = new Session (graph);
	long START = System.currentTimeMillis();
	for ( int i = 0; i < 10000; i++ ) {
	    Tensor tensor = session.runner()
		.fetch ("z")
		.feed( "x", Tensor.create( 3.0 + i, Double.class ) )
		.feed( "y", Tensor.create( 6.0, Double.class ) )
		.run()
		.get (0).expect (Double.class);
	    if ( i == 0 )
		System.out.println (tensor.doubleValue());
	}
	long END = System.currentTimeMillis();
	System.out.println ("" + (END - START) + "ms");
    }

    public static void main( String args[] ) {
	try {
	    sample();
	}
	catch ( Exception e ) {
	    e.printStackTrace();
	}
    }
}
