package com.vartanian.titanic;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.transform.transform.string.ReplaceEmptyStringTransform;
import org.datavec.api.transform.transform.string.ReplaceStringTransform;
import org.datavec.api.transform.transform.string.StringMapTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
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
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class App {

    public static void main(String[] args) throws IOException, InterruptedException {
        int labelIndex = 0;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 1300;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsInteger("PassengerId", "Survived", "Pclass")
                .addColumnsString("Name", "Sex")
                .addColumnDouble("Age")
                .addColumnsInteger("SibSp", "Parch")
                .addColumnsString("Ticket")
                .addColumnDouble("Fare")
                .addColumnString("Cabin")
                .addColumnCategorical("Embarked", Arrays.asList("S","C","Q"))
                .build();
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                //Let's remove some column we don't need
                .removeColumns("Name","Cabin")

//                //Now, suppose we only want to analyze transactions involving merchants in USA or Canada. Let's filter out
//                // everything except for those countries.
//                //Here, we are applying a conditional filter. We remove all of the examples that match the condition
//                // The condition is "MerchantCountryCode" isn't one of {"USA", "CAN"}
//                .filter(new ConditionFilter(
//                        new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, new HashSet<>(Arrays.asList("USA","CAN")))))

                //Let's suppose our data source isn't perfect, and we have some invalid data: negative dollar amounts that we want to replace with 0.0
                //For positive dollar amounts, we don't want to modify those values
                //Use the ConditionalReplaceValueTransform on the "TransactionAmountUSD" column:
                .transform(new StringMapTransform()ReplaceInvalidWithIntegerTransform("Embarked"))
//        .conditionalReplaceValueTransform(
//                        "TransactionAmountUSD",     //Column to operate on
//                        new DoubleWritable(0.0),    //New value to use, when the condition is satisfied
//                        new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) //Condition: amount < 0.0
//
//                //Finally, let's suppose we want to parse our date/time column in a format like "2016/01/01 17:50.000"
//                //We use JodaTime internally, so formats can be specified as follows: http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html
//                .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
//
//                //However, our time column ("DateTimeString") isn't a String anymore. So let's rename it to something better:
//                .renameColumn("DateTimeString", "DateTime")
//
//                //At this point, we have our date/time format stored internally as a long value (Unix/Epoch format): milliseconds since 00:00.000 01/01/1970
//                //Suppose we only care about the hour of the day. Let's derive a new column for that, from the DateTime column
//                .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime")
//                        .addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay())
//                        .build())
//
//                //We no longer need our "DateTime" column, as we've extracted what we need from it. So let's remove it
//                .removeColumns("DateTime")

                //We've finished with the sequence of operations we want to do: let's create the final TransformProcess object
                .build();


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);








        RecordReader recordReader = new CSVRecordReader(0, ',', '"');
        recordReader.initialize(new FileSplit(new ClassPathResource("train.csv").getFile()));
        //Process the data:
        List<List<Writable>> originalData = new ArrayList<List<Writable>>();
        while(recordReader.hasNext()){
            originalData.add(recordReader.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);

        for(List<Writable> line : processedData){
            System.out.println(line);
        }


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
