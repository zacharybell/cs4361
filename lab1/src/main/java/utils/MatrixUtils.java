package utils;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Collections;
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

    public static RealMatrix normalize(RealMatrix matrix) {
        double[][] data = matrix.transpose().getData();

        for (double[] e : data) {
            normalize(e);
        }

        return (new Array2DRowRealMatrix(data)).transpose();
    }

    public static void normalize(double[] vector) {
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

    private static void swap(double[][] array, int i, int j) {
        double[] temp;
        temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
