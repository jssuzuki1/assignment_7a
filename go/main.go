package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/e-XpertSolutions/go-iforest/iforest"
	"github.com/petar/GoMNIST"
)

func main() {

	// Record start time
	startTime := time.Now()

	// Loop 100 times
	for iteration := 0; iteration < 100; iteration++ {

		//Loads data into [][]float64
		train, _, err := GoMNIST.Load("..\\data")
		if err != nil {
			log.Fatal(err)
		}

		// Create dataframes for images and labels.
		images := make([][]float64, len(train.Images))
		labels := make([]int, len(train.Images))

		for i := 0; i < len(train.Images); i++ {
			images[i] = make([]float64, len(train.Images[0]))
			for p := range train.Images[0] {
				//Integer pixel values 0-255 to float64 0.00-255.00 for iforest
				images[i][p] = float64(train.Images[i][p])
				labels[i] = int(train.Labels[i])

			}
		}

		// input parameters
		treesNumber := 100
		subsampleSize := 256
		outliersRatio := 0.0001

		// model initialization
		forest := iforest.NewForest(treesNumber, subsampleSize, outliersRatio)

		// Train on the images data set
		forest.Train(images)

		// Test function is necessary to generate Anomaly Scores for Each Sample
		forest.Test(images)

		// format of anomalyScores is map[int]float64
		anomalyScores := forest.AnomalyScores

		// Create a dataframe called "AnomalyScores" that has the length of all of the AnomalyScores + 1
		// The +1 is for the header.
		var scores = make([][]string, len(anomalyScores)+1)

		// This for loop goes through every record of "scores" and populates it.
		for i := 0; i < len(scores); i++ {
			scores[i] = make([]string, 2)
			// The first row is the header
			if i == 0 {
				scores[0][0] = "RowID"
				scores[0][1] = "Scores"
			}

			// The second row onward is populated with anomaly scores.
			// Subtract 0.5 since the anomaly scores from iforest are normalized around 0.5 as opposed to 0, allowing for negative values.
			// Because i == 0 is reserved for the header, when i = 1, we want to select the first row of the AnomalyScores, which is the 0th element.
			if i != 0 {
				//Anomaly Scores
				score := 0.5 - anomalyScores[i-1]
				scores[i][0] = fmt.Sprintf("%d", i-1)
				scores[i][1] = fmt.Sprintf("%f", score)
			}

		}

		//Export
		file, _ := os.Create("../results/go_scores.csv")
		w := csv.NewWriter(file)
		w.WriteAll(scores)
	}
	// Record end time
	endTime := time.Now()

	// Calculate and print total duration
	totalDuration := endTime.Sub(startTime)
	avg_dur := totalDuration / 100

	fmt.Println("Total time:", totalDuration)

	// Output the go run time to a text file in the results folder.
	file, _ := os.Create("../results/go_time.txt")
	_, _ = file.WriteString(avg_dur.String())

}
