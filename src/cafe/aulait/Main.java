package cafe.aulait;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public class Main {
    public static final int N = 10;
    public static final float LEARNING_RATE = 0.1f;
    public static final String WEIGHT_VARIALBE_NAME = "weight";
    
    


    public static void main( String args[] ) {
	System.out.println ("Hello, World!");
	Graph g = new Graph();
	g.opBuilder( "Variable", "x" )
	    .setAttr( "dtype", DataType.DOUBLE )
	    .setAttr( "shape", Shape.scalar() );
    }
}
