package org.myorg;
import java.io.IOException;
import java.util.Arrays;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class KMeansClustering {

    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {

        private float[] centroids;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            String[] means = conf.getStrings("centroids");
            centroids = new float[means.length];
            for (int i = 0; i < means.length; i++) {
                centroids[i] = Float.parseFloat(means[i]);
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // Convert input value to float
            float dataPoint = Float.parseFloat(value.toString());

            // Find the nearest centroid
            float minDistance = Float.MAX_VALUE;
            int nearestCentroidIndex = 0;
            for (int i = 0; i < centroids.length; i++) {
                float distance = Math.abs(dataPoint - centroids[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroidIndex = i;
                }
            }

            // Emit nearest centroid index as key and data point value as the value
            context.write(new IntWritable(nearestCentroidIndex), new Text(Float.toString(dataPoint)));
        }
    }


    public static class FinalMapper extends Mapper<Object, Text, Text, Text> {

        private float[] centroids;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            String[] means = conf.getStrings("centroids");
            centroids = new float[means.length];
            for (int i = 0; i < means.length; i++) {
                    centroids[i] = Float.parseFloat(means[i]);
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // Convert input value to float
            float dataPoint = Float.parseFloat(value.toString());

            // Find the nearest centroid
            float minDistance = Float.MAX_VALUE;
            float nearestCentroid = centroids[0];
            for (float centroid : centroids) {
                    float distance = Math.abs(dataPoint - centroid);
                    if (distance < minDistance) {
                            minDistance = distance;
                            nearestCentroid = centroid;
                    }
            }

            // Emit data point value as key and centroid value as the value
            context.write(new Text(Float.toString(dataPoint)), new Text(Float.toString(nearestCentroid)));
        }
    }


    public static class KMeansReducer extends Reducer<IntWritable, Text, NullWritable, Text> {

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            float sum = 0;
            int count = 0;
            for (Text value : values) {
                sum += Float.parseFloat(value.toString());
                count++;
            }
            // Calculate the new mean for the centroid
            float newMean = sum / count;

            // Emit the new mean
            context.write(NullWritable.get(), new Text(Float.toString(newMean)));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: KMeansClustering <inputPath> <outputPath> <K> <initialMeans> <maxIterations>");
            System.exit(1);
        }

        String inputPath = args[0];
        String outputPath = args[1];
        int K = Integer.parseInt(args[2]);
        String[] initialMeans = args[3].split(",");
        int maxIterations = Integer.parseInt(args[4]);

        if (initialMeans.length != K) {
            System.err.println("Number of initial means must match K");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        conf.setInt("K", K);
        conf.setStrings("centroids", initialMeans);

        // Run the job iteratively for the specified number of iterations
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            Job job = Job.getInstance(conf, "KMeans Clustering - Iteration " + iteration);
            job.setJarByClass(KMeansClustering.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);
            job.setOutputKeyClass(NullWritable.class);
            job.setOutputValueClass(Text.class);
            job.setOutputFormatClass(TextOutputFormat.class);
            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(outputPath + iteration));

            job.waitForCompletion(true);

            // Update initialMeans with the new means obtained from the output of the reducer
            String[] newMeans = readMeansFromOutput(job, K);
            conf.setStrings("centroids", newMeans);
        }

        // Run the final job to assign data points to their nearest centroids
        Job finalJob = Job.getInstance(conf, "Final KMeans Clustering");
        finalJob.setJarByClass(KMeansClustering.class);
        finalJob.setMapperClass(FinalMapper.class);
        finalJob.setOutputKeyClass(Text.class);
        finalJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(finalJob, new Path(inputPath));
        FileOutputFormat.setOutputPath(finalJob, new Path(outputPath + "final"));

        System.exit(finalJob.waitForCompletion(true) ? 0 : 1);
    }

    private static String[] readMeansFromOutput(Job job, int K) throws IOException {
        String[] means = new String[K];
        Arrays.fill(means, "0.0"); // Initialize with default value

        Path outputPath = FileOutputFormat.getOutputPath(job);
        FileSystem fs = outputPath.getFileSystem(job.getConfiguration());
        try (FSDataInputStream inputStream = fs.open(new Path(outputPath, "part-r-00000"))) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
                String line;
                int index = 0;
                while ((line = reader.readLine()) != null && index < K) {
                        means[index++] = line;
                }
            }
        }
        return means;
    }
}