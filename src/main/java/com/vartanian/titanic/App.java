package com.vartanian.titanic;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class App {

    public static void main(String[] args) throws IOException, InterruptedException {
        int labelIndex = 0;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 1300;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        RecordReader recordReader = new CSVRecordReader(1, ',', '"');
        recordReader.initialize(new FileSplit(new ClassPathResource("train.csv").getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData); //Apply normalization to the test data. This is using statistics calculated from the *training* set

        //Create the model
        int nIn = 6;
        int nOut = 32;
        int numIterations = 500;
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs())
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                .layer(1, new DenseLayer.Builder().nIn(nOut).nOut(nOut).build())
                .layer(2, new OutputLayer.Builder().nIn(nOut).nOut(2).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(1));
        for(int i=0; i < numIterations; i++) {
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

        INDArray dicaprio = ( Nd4j.create( new double[] { 3, 1, 19, 0 , 0 , 25.0} ));
        INDArray winslet = ( Nd4j.create( new double[] { 1, 0, 19, 1 , 2 , 75.0} ));
        normalizer.transform( dicaprio );
        normalizer.transform( winslet );

        String[] classes = { "Dead" , "Survived" };
        int survivedIndex = 1;

        int[] result = model.predict( dicaprio );
        System.out.println( "DiCaprio   Surviving Rate: " + model.output(dicaprio).getColumn(survivedIndex) + "  class: "+ classes[result[0]] );

        result = model.predict( winslet );
        System.out.println( "Winslet    Surviving Rate: " +  model.output(winslet).getColumn(survivedIndex) + "  class: "+ classes[result[0]] );


    }

}
