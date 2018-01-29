package functional;

import models.MostCommonClassifier;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jetbrains.annotations.NotNull;
import utils.MatrixUtils;

import java.io.*;
import java.util.LinkedList;
import java.util.List;

public class ClassifierTest {

    public static void main(String[] args) {

        double splitRatio = 0.7;

        RealMatrix data = new Array2DRowRealMatrix(readCSV("lab1/src/test/resources/iris_numlabel.txt"));

        data = MatrixUtils.shuffle(data, 99);

        // split
        RealMatrix X = data.getSubMatrix(0, data.getRowDimension() - 1, 0, data.getColumnDimension() - 2);
        RealVector y = data.getColumnVector(data.getColumnDimension() - 1);

        X = MatrixUtils.normalize(X);

        // split
        RealMatrix XTrain = X.getSubMatrix(0, (int)(X.getRowDimension() * splitRatio) - 1, 0, X.getColumnDimension() - 1);
        RealVector yTrain = y.getSubVector(0, (int)(y.getDimension() * splitRatio));
        RealMatrix XTest = X.getSubMatrix((int)(X.getRowDimension() * splitRatio), X.getRowDimension() - 1, 0, X.getColumnDimension() - 1);
        RealVector yTest = y.getSubVector((int)(y.getDimension() * splitRatio), y.getDimension() - yTrain.getDimension());


        // Most common classifier
        MostCommonClassifier mcc = new MostCommonClassifier();
        mcc.fit(XTrain, yTrain);
        RealVector yPredict = mcc.predict(XTest);



    }

    /**
     * Read in a CSV file and parse into a double[][]. The rows of the CSV file coincide with the first index of the
     * array and the columns with the second.
     *
     * @param path the relative or absolute path to the csv file
     * @return a double[][] containing all the values from the file
     */
    private static double[][] readCSV(@NotNull String path) {

        double[][] data = null;
        List<double[]> dataLL = new LinkedList<>();
        Reader in;

        try {
            in = new FileReader(path);
            Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse(in);
            for (CSVRecord record : records) {
                double[] row = new double[record.size()];
                for (int i = 0; i < row.length; i++) {
                    row[i] = Double.valueOf(record.get(i));
                }
                dataLL.add(row);
            }

            data = new double[dataLL.size()][];
            dataLL.toArray(data);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return data;
    }
}
