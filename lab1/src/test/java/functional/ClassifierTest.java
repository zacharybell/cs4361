package functional;

import models.MostCommonClassifier;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jetbrains.annotations.NotNull;
import preprocessing.MatrixUtils;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class ClassifierTest {

    public static void main(String[] args) {

//        RealMatrix data = new Array2DRowRealMatrix(readCSV("lab1/src/test/resources/iris_numlabel.txt"));
//        RealMatrix X = data.getSubMatrix(0, data.getRowDimension() - 1, 0, data.getColumnDimension() - 2);
//        RealVector y = data.getColumnVector(data.getColumnDimension() - 1);
//
//        MostCommonClassifier mcc = new MostCommonClassifier();
//        mcc.fit(X, y);
//        RealVector g = mcc.predict(X);
//
//        System.out.println(g);

//        RealMatrix matrix = new Array2DRowRealMatrix(new double[][] {
//                {1.0, 1.0},
//                {2.0, 2.0},
//                {3.0, 3.0},
//                {4.0, 4.0},
//                {5.0, 5.0},
//                {6.0, 6.0},
//        });
//
//        System.out.println(MatrixUtils.shuffle(matrix, 5));
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
