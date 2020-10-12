package cafe.aulait;

import org.tensorflow.Tensor;
import java.util.Arrays;

public class TensorCreation {
    public static void main( String args[] ) {
	Tensor<Integer> rank0Tensor = Tensor.create( 42, Integer.class );
	System.out.println ("---- Scalar ----");
	System.out.println ("DataType: " + rank0Tensor.dataType().name());
	System.out.println ("Rank: " + rank0Tensor.shape().length);
	System.out.println("Shape: " + Arrays.toString (rank0Tensor.shape()));
    }
}
