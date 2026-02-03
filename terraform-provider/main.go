// FHE-GBDT Terraform Provider
// Infrastructure-as-Code for FHE-GBDT-Serving

package main

import (
	"context"
	"flag"
	"log"

	"github.com/hashicorp/terraform-plugin-framework/providerserver"
	"github.com/fhe-gbdt-serving/terraform-provider/internal/provider"
)

var (
	version = "0.1.0"
)

func main() {
	var debug bool

	flag.BoolVar(&debug, "debug", false, "set to true to run the provider with support for debuggers")
	flag.Parse()

	opts := providerserver.ServeOpts{
		Address: "registry.terraform.io/fhe-gbdt/fhe-gbdt",
		Debug:   debug,
	}

	err := providerserver.Serve(context.Background(), provider.New(version), opts)
	if err != nil {
		log.Fatal(err.Error())
	}
}
