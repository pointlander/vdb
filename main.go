// Copyright 2024 The VDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
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
}

func dot(a []float64, b []float64) float64 {
	sum := 0.0
	for i, v := range a {
		sum += v * b[i]
	}
	return sum
}

func dotT(a []float64, b []Vector, col int) float64 {
	sum := 0.0
	for i, v := range a {
		sum += v * b[i].V[col]
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

// SelfEntropy computes the self entropy of X
func SelfEntropy(x []Vector) []float64 {
	cols, rows := len(x[0].V), len(x)
	entropies, values, results := make([]float64, cols), make([]float64, rows), make([]float64, 0, rows)
	for _, k := range x {
		for j, q := range x {
			values[j] = dot(k.V, q.V)
		}
		softmax(values)

		for j := 0; j < cols; j++ {
			entropies[j] = dotT(values, x, j)
		}
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		results = append(results, -entropy)
	}
	return results
}

func main() {

}
