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
	vdb := make([]Vector, len(datum.Fisher))
	for i, v := range datum.Fisher {
		measures := make([]float64, len(v.Measures))
		copy(measures, v.Measures)
		vdb[i].V = measures
		vdb[i].Label = v.Label
	}
	entropy := SelfEntropy(vdb)
	for i, e := range entropy {
		vdb[i].Entropy = e
	}
	sort.Slice(vdb, func(i, j int) bool {
		return vdb[i].Entropy < vdb[j].Entropy
	})
	last, count := "", 0
	for i, v := range vdb {
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
