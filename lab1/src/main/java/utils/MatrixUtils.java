package utils;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

/**
 * Utilities supporting machine learning implementations.
 */
public class MatrixUtils {

    /**
     * Computes the accuracy between two vectors.
     *
     * @param AVector
     * @param BVector
     * @return a double between 0.0 and 1.0
     */
    public static double accuracy(@NotNull RealVector AVector, @NotNull RealVector BVector) {
        double epsilon = 0.000001;

        double[] A = AVector.toArray();
        double[] B = BVector.toArray();

        if (A.length != B.length) throw new IllegalArgumentException("Both vectors need to be of the same length!");
        if (A.length == 0) throw new IllegalArgumentException("Vectors cannot be of length 0!");

        double error = 0;

        for (int i = 0; i < A.length; i++) {
            if (Math.abs(Double.compare(A[i], B[i])) < epsilon) error += 1;
        }

        return error / A.length;
    }

    /**
     * Shuffles a matrix by row.
     *
     * @param matrix
     * @param seed the random seed
     * @return a shuffled real matrix
     */
    public static RealMatrix shuffle(@NotNull RealMatrix matrix, int seed) {
        Random random = new Random(seed);

        double[][] data = matrix.getData();

        for (int i = 0; i < data.length; i++) {
            swap(data, i, random.nextInt(data.length));
        }

        return new Array2DRowRealMatrix(data);
    }

    /**
     * Normalizes a matrix by column. The normalization equation used is N(x) = (x - min(x)) / (max(x) - min(x)).
     *
     * @param matrix
     * @return a normalized matrix
     */
    public static RealMatrix normalize(@NotNull RealMatrix matrix) {
        double[][] data = matrix.transpose().getData();

        for (double[] e : data) {
            normalize(e);
        }

        return (new Array2DRowRealMatrix(data)).transpose();
    }

    private static void normalize(@NotNull double[] vector) {
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;

        for (int i = 0; i < vector.length; i++) {
            if (max < vector[i]) {
                max = vector[i];
            }
            if (min > vector[i]) {
                min = vector[i];
            }
        }

        for (int i = 0; i < vector.length; i++) {
            vector[i] = (vector[i] - min) / (max - min);
        }
    }

    private static void swap(@NotNull double[][] array, int i, int j) {
        double[] temp;
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
