import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object alsrecommendation {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("ALS_Movie_Recommendation")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "C:\\sql")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val ratingsFile = "./data/ratings.csv"
    val df1 = spark.read.format("com.databricks.spark.csv")
      .option("header", "true").load(ratingsFile)

    val ratingsDF = df1.select(df1.col("userId"), df1.col("movieId"),
      df1.col("rating"), df1.col("timestamp"))
    ratingsDF.show(false)

    val moviesFile = "./data/movies.csv"
    val df2 = spark.read.format("com.databricks.spark.csv")
      .option("header", "true").load(moviesFile)

    val moviesDF = df2.select(df2.col("movieId"), df2.col("title"),
      df2.col("genres"))
    moviesDF.show(false)

    ratingsDF.createOrReplaceTempView("ratings")
    moviesDF.createOrReplaceTempView("movies")


    // Explore and query with Spark DataFrames
    val numRatings = ratingsDF.count()
    val numUsers = ratingsDF.select(ratingsDF.col("userId")).distinct().count()
    val numMovies = ratingsDF.select(ratingsDF.col("movieId")).distinct().count()
    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies
      + " movies.")

    // Get the max, min ratings along with the count of users who have rated a movie.
    val results = spark.sql(
      "SELECT movies.title, movierates.maxr, movierates.minr, "
        + "movierates.cntu FROM (SELECT ratings.movieId,max(ratings.rating) AS maxr, "
        + "MIN(ratings.rating) AS minr, COUNT(distinct userId) AS cntu "
        + "FROM ratings GROUP BY ratings.movieId) movierates "
        + "JOIN movies ON movierates.movieId = movies.movieId "
        + "ORDER BY movierates.cntu DESC ")

    results.show(false)

    // Show the top 10 most-active users and how many times they rated a movie
    val mostActiveUsersSchemaRDD = spark.sql(
      "SELECT ratings.userId, COUNT(*) AS ct FROM ratings "
        + "GROUP BY ratings.userId ORDER BY ct DESC LIMIT 10 ")

    mostActiveUsersSchemaRDD.show(false)

    // Find the movies that user 610 rated higher than 4
    val results2 = spark.sql(
      "SELECT ratings.userId, ratings.movieId, ratings.rating, movies.title "
        + "FROM ratings JOIN movies "
        + "ON movies.movieId = ratings.movieId "
        + "WHERE ratings.userId = 610 AND ratings.rating > 4 ")

    results2.show(false)


    // Randomly split ratings RDD into training data RDD (70%) and test data RDD (30%)
    val splits = ratingsDF.randomSplit(Array(0.70, 0.30), seed = 12345L)
    val (trainingData, testData) = (splits(0), splits(1))

    val numTraining = trainingData.count()
    val numTest = testData.count()
    println("Training: " + numTraining + " Test: " + numTest)

    val ratingsRDD = trainingData.rdd.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
    })

    val testRDD = testData.rdd.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
    })


    /*
		 * Using ALS with the Movie Ratings Data: build a ALS user product matrix model with
		 * rank = 10, iterations = 10. The training is done via the Collaborative Filtering
		 * algorithm Alternating Least Squares (ALS). Essentially this technique predicts missing
		 * ratings for specific users for specific movies based on ratings for those movies from
		 * other users who did similar ratings for other movies.
		 */

    val model = new ALS()
      .setIterations(10)
      .setBlocks(-1)
      .setAlpha(1.0)
      .setLambda(0.10)
      .setRank(10)
      .setSeed(22L)
      .setImplicitPrefs(false)
      .run(ratingsRDD)


    /*
		 * Evaluating the Model:
		 * In order to verify the quality of the models Root Mean Squared Error (RMSE) is used to
		 * measure the differences between values predicted by a model and the values actually
		 * observed. The smaller the calculated error, the better the model. In order to test the
		 * quality of the model, the test data is used (which was split above). RMSE is a good
		 * measure of accuracy, but only to compare forecasting errors of different models for a
		 * particular variable and not between variables, as it is scale-dependent.
		 */

    val rmseTest = computeRMSE(model, testRDD, true)
    println("Test RMSE = " + rmseTest)

    // Making Predictions. Get the top 5 movie predictions for user 610
    println("Rating:(UserID, MovieID, Rating)")
    println("----------------------------------")
    val topRecsForUser = model.recommendProducts(610, 5)
    for (rating <- topRecsForUser) {
      println(rating.toString())
    }
    println("----------------------------------")

    // Movie recommendation for a specific user. Get the top 5 movie predictions for user 610
    println("Recommendations: (MovieId => Rating)")
    println("----------------------------------")
    val recommendationsUser = model.recommendProducts(610, 5)
    recommendationsUser.map(rating => (rating.product, rating.rating)).foreach(println)
    println("----------------------------------")

    spark.stop()
  }

  // Compute RMSE to evaluate the model. Less the RMSE better the model and it's prediction power.
  def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating],
                  implicitPrefs: Boolean): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating))).values
    if (implicitPrefs) {
      println("(Prediction, Rating)")
      println(predictionsAndRatings.take(5).mkString("\n"))
    }
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }
}
