package preprocessing;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class MatrixUtils {
    public static RealMatrix shuffle(RealMatrix matrix, int seed) {
        Random random = new Random(seed);

        double[][] data = matrix.getData();

        for (int i = 0; i < data.length; i++) {
            swap(data, i, random.nextInt(data.length));
        }

        return new Array2DRowRealMatrix(data);
    }

    private static void swap(double[][] array, int i, int j) {
        double[] temp;
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
