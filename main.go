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

func dot(a []float64, b []float64) float64 {
	sum := 0.0
	for i, v := range a {
		sum += v * b[i]
	}
	return sum
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

// SelfEntropy computes the self entropy of X
func (v VDB) SelfEntropy() []float64 {
	cols, rows := v.Width, len(v.Rows)
	entropies, values, results := make([]float64, cols), make([]float64, rows), make([]float64, 0, rows)
	for _, k := range v.Rows {
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
		results = append(results, -entropy)
	}
	return results
}

func main() {

}
