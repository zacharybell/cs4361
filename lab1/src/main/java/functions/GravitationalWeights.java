package functions;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.function.Function;

/**
 * Lambda equation that takes a real matrix and applies gravitational weights to determine distance between the two
 * row vectors within the matrix. Vectors that are very similar will have a higher weight. The equation used utilizes
 * a 1/r^2 method to compute.
 */
public class GravitationalWeights implements Function<RealMatrix, Double> {
    @Override
    public Double apply(RealMatrix realMatrix) {

        if (realMatrix.getRowDimension() != 2) throw new IllegalArgumentException("RealMatrix dimension need to be 2XN!");

        double[] rowA = realMatrix.getRow(0);
        double[] rowB = realMatrix.getRow(1);

        if (rowA.length != rowB.length) throw new IllegalArgumentException("Both rows need to be of the same length!");

        double weightedDistance = 0;
        for (int i = 0; i < rowA.length; i++) {
            weightedDistance += (1 / Math.pow(rowA[i] - rowB[i] + 0.000001, 2));
        }

        return weightedDistance;
    }
}
