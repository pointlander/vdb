// Copyright 2024 The VDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"strconv"

	"github.com/pointlander/datum/mnist"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

// Vector is a vector
type Vector struct {
	V       []float64
	Entropy float64
	Label   string
	Type    int
}

// VDB is a vector database
type VDB struct {
	Width int
	Rows  []Vector
}

// NewVDB makes a new vector database
func NewVDB(width int) VDB {
	return VDB{
		Width: width,
	}
}

func dotT(a []float64, b VDB, col int) float64 {
	sum := 0.0
	for i, v := range a {
		sum += v * b.Rows[i].V[col]
	}
	return sum
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// Slice takes a slice of the vdb
func (v VDB) Slice(begin, end int) VDB {
	return VDB{
		Width: v.Width,
		Rows:  v.Rows[begin:end],
	}
}

// SelfEntropy computes the self entropy of X
func (v VDB) SelfEntropy() {
	cols, rows := v.Width, len(v.Rows)
	entropies, values := make([]float64, cols), make([]float64, rows)
	for i, k := range v.Rows {
		for j, q := range v.Rows {
			values[j] = dot(k.V, q.V)
		}
		softmax(values)

		for j := 0; j < cols; j++ {
			entropies[j] = dotT(values, v, j)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		v.Rows[i].Entropy = -entropy
	}
}

// Rainbow computes the rainbow algorithm
func (v VDB) Rainbow(iterations int) {
	cpus, done := runtime.NumCPU(), make(chan bool, 8)
	process := func(begin, end int) {
		s := v.Slice(begin, end)
		s.SelfEntropy()
		done <- true
	}
	for j := 0; j < iterations; j++ {
		i, flight := 0, 0
		for i < len(v.Rows)-100 && flight < cpus {
			go process(i, i+100)
			i += 100
			flight++
		}
		for i < len(v.Rows)-100 {
			<-done
			flight--

			go process(i, i+100)
			i += 100
			flight++
		}
		for k := 0; k < flight; k++ {
			<-done
		}
		sort.Slice(v.Rows, func(i, j int) bool {
			return v.Rows[i].Entropy < v.Rows[j].Entropy
		})
	}
}

func main() {
	datum, err := mnist.Load()
	if err != nil {
		panic(err)
	}

	fmt.Println("loading database")
	db := NewVDB(datum.Train.Width * datum.Train.Height)
	for i, image := range datum.Train.Images {
		vector := make([]float64, len(image))
		sum := 0.0
		for j, value := range image {
			vector[j] = float64(value)
			sum += float64(value)
		}
		for i, v := range vector {
			vector[i] = v / sum
		}
		db.Rows = append(db.Rows, Vector{
			V:     vector,
			Label: strconv.Itoa(int(datum.Train.Labels[i])),
		})
	}
	for i, image := range datum.Test.Images {
		vector := make([]float64, len(image))
		sum := 0.0
		for j, value := range image {
			vector[j] = float64(value)
			sum += float64(value)
		}
		for i, v := range vector {
			vector[i] = v / sum
		}
		db.Rows = append(db.Rows, Vector{
			V:     vector,
			Label: strconv.Itoa(int(datum.Test.Labels[i])),
			Type:  1,
		})
	}
	fmt.Println("calculating entropy")
	db.Rainbow(32)
	fmt.Println("testing db")
	correct := 0
	for i, row := range db.Rows {
		if row.Type == 1 {
			index := i + 1
			if row.Label == db.Rows[index].Label {
				correct++
			}
		}
	}
	fmt.Println(correct)
}
