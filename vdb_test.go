// Copyright 2024 The VDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sort"
	"testing"

	"github.com/pointlander/datum/iris"
)

func TestSelfEntropy(t *testing.T) {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	vdb := NewVDB(4)
	vdb.Rows = make([]Vector, len(datum.Fisher))
	for i, v := range datum.Fisher {
		measures := make([]float64, len(v.Measures))
		copy(measures, v.Measures)
		vdb.Rows[i].V = measures
		vdb.Rows[i].Label = v.Label
	}
	entropy := vdb.SelfEntropy()
	for i, e := range entropy {
		vdb.Rows[i].Entropy = e
	}
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
