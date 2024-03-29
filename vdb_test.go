// Copyright 2024 The VDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"sort"
	"strconv"
	"testing"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/datum/mnist"
)

func TestSelfEntropy(t *testing.T) {
	datum, err := iris.Load()
	if err != nil {
		t.Fatal(err)
	}
	vdb := NewVDB(4)
	vdb.Rows = make([]Vector, len(datum.Fisher))
	for i, v := range datum.Fisher {
		measures := make([]float64, len(v.Measures))
		copy(measures, v.Measures)
		vdb.Rows[i].V = measures
		vdb.Rows[i].Label = v.Label
	}
	vdb.SelfEntropy()
	sort.Slice(vdb.Rows, func(i, j int) bool {
		return vdb.Rows[i].Entropy < vdb.Rows[j].Entropy
	})
	last, count := "", 0
	for i, v := range vdb.Rows {
		if last != v.Label {
			count++
		}
		last = v.Label
		t.Log(i, v.Label, v.Entropy)
	}
	t.Log(count)
	if count != 7 {
		t.Fatal("count should be 7")
	}
}

func TestMNIST(t *testing.T) {
	datum, err := mnist.Load()
	if err != nil {
		t.Fatal(err)
	}
	key := NewVDB(datum.Train.Width * datum.Train.Height)
	query, count := uint8(0), 0
	rng := rand.New(rand.NewSource(1))
	for query < 10 {
		i := rng.Intn(len(datum.Train.Images))
		image := datum.Train.Images[i]
		if datum.Train.Labels[i] == query {
			t.Log(query)
			vector := make([]float64, len(image))
			for i, value := range image {
				vector[i] = float64(value)
			}
			key.Rows = append(key.Rows, Vector{
				V:     vector,
				Label: strconv.Itoa(int(query)),
			})
			count++
			if count == 5 {
				query++
				count = 0
			}
		}
	}
	t.Log("build database")
	key.Rows = append(key.Rows, Vector{})
	db := NewVDB(datum.Train.Width * datum.Train.Height)
	for i, image := range datum.Train.Images {
		vector := make([]float64, len(image))
		for j, value := range image {
			vector[j] = float64(value)
		}
		key.Rows[len(key.Rows)-1] = Vector{
			V:     vector,
			Label: strconv.Itoa(int(datum.Train.Labels[i])),
		}
		key.SelfEntropy()
		db.Rows = append(db.Rows, key.Rows[len(key.Rows)-1])
	}
	t.Log("sort database")
	sort.Slice(db.Rows, func(i, j int) bool {
		return db.Rows[i].Entropy < db.Rows[j].Entropy
	})
	t.Log("test database")
	correct := 0
	for i, image := range datum.Test.Images {
		vector := make([]float64, len(image))
		for j, value := range image {
			vector[j] = float64(value)
		}
		key.Rows[len(key.Rows)-1] = Vector{
			V:     vector,
			Label: strconv.Itoa(int(datum.Test.Labels[i])),
		}
		key.SelfEntropy()
		query := key.Rows[len(key.Rows)-1].Entropy
		index := sort.Search(len(db.Rows), func(i int) bool {
			return db.Rows[i].Entropy >= query
		})
		labels := make([]int, 10)
		for j := index; j < index+10 && j < len(db.Rows); j++ {
			label, _ := strconv.Atoi(db.Rows[j].Label)
			labels[label]++
		}
		max, label := 0, 0
		for j, v := range labels {
			if v > max {
				max, label = v, j
			}
		}
		if index < len(db.Rows) {
			if key.Rows[len(key.Rows)-1].Label == strconv.Itoa(label) {
				correct++
			}
		}
	}
	t.Log(correct, len(datum.Test.Images))
}
