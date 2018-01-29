package functions;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.function.Function;

public class GravitationalWeights implements Function<RealMatrix, Double> {
    @Override
    public Double apply(RealMatrix realMatrix) {

        if (realMatrix.getRowDimension() != 2) throw new IllegalArgumentException("RealMatrix dimension need to be 2XN!");

        double[] rowA = realMatrix.getRow(0);
        double[] rowB = realMatrix.getRow(1);

        if (rowA.length != rowB.length) throw new IllegalArgumentException("Both rows need to be of the same length!");

        double weightedDistance = 0;
        for (int i = 0; i < rowA.length; i++) {
            weightedDistance += (1 / Math.pow(rowA[i] - rowB[i], 2));
        }

        return weightedDistance;
    }
}
