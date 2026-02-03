// FHE-GBDT Provider Implementation

package provider

import (
	"context"
	"os"

	"github.com/hashicorp/terraform-plugin-framework/datasource"
	"github.com/hashicorp/terraform-plugin-framework/path"
	"github.com/hashicorp/terraform-plugin-framework/provider"
	"github.com/hashicorp/terraform-plugin-framework/provider/schema"
	"github.com/hashicorp/terraform-plugin-framework/resource"
	"github.com/hashicorp/terraform-plugin-framework/types"
)

// Ensure FHEGBDTProvider satisfies various provider interfaces.
var _ provider.Provider = &FHEGBDTProvider{}

// FHEGBDTProvider defines the provider implementation.
type FHEGBDTProvider struct {
	version string
}

// FHEGBDTProviderModel describes the provider data model.
type FHEGBDTProviderModel struct {
	Endpoint types.String `tfsdk:"endpoint"`
	APIKey   types.String `tfsdk:"api_key"`
	TenantID types.String `tfsdk:"tenant_id"`
	Region   types.String `tfsdk:"region"`
}

func New(version string) func() provider.Provider {
	return func() provider.Provider {
		return &FHEGBDTProvider{
			version: version,
		}
	}
}

func (p *FHEGBDTProvider) Metadata(ctx context.Context, req provider.MetadataRequest, resp *provider.MetadataResponse) {
	resp.TypeName = "fhegbdt"
	resp.Version = p.version
}

func (p *FHEGBDTProvider) Schema(ctx context.Context, req provider.SchemaRequest, resp *provider.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Interact with FHE-GBDT-Serving platform for privacy-preserving ML inference.",
		Attributes: map[string]schema.Attribute{
			"endpoint": schema.StringAttribute{
				Description: "The FHE-GBDT API endpoint. Can also be set via FHE_GBDT_ENDPOINT env var.",
				Optional:    true,
			},
			"api_key": schema.StringAttribute{
				Description: "The API key for authentication. Can also be set via FHE_GBDT_API_KEY env var.",
				Optional:    true,
				Sensitive:   true,
			},
			"tenant_id": schema.StringAttribute{
				Description: "The tenant/organization ID. Can also be set via FHE_GBDT_TENANT_ID env var.",
				Optional:    true,
			},
			"region": schema.StringAttribute{
				Description: "The default region for resources. Can also be set via FHE_GBDT_REGION env var.",
				Optional:    true,
			},
		},
	}
}

func (p *FHEGBDTProvider) Configure(ctx context.Context, req provider.ConfigureRequest, resp *provider.ConfigureResponse) {
	var config FHEGBDTProviderModel

	resp.Diagnostics.Append(req.Config.Get(ctx, &config)...)
	if resp.Diagnostics.HasError() {
		return
	}

	// Set defaults from environment variables
	endpoint := os.Getenv("FHE_GBDT_ENDPOINT")
	apiKey := os.Getenv("FHE_GBDT_API_KEY")
	tenantID := os.Getenv("FHE_GBDT_TENANT_ID")
	region := os.Getenv("FHE_GBDT_REGION")

	if !config.Endpoint.IsNull() {
		endpoint = config.Endpoint.ValueString()
	}
	if !config.APIKey.IsNull() {
		apiKey = config.APIKey.ValueString()
	}
	if !config.TenantID.IsNull() {
		tenantID = config.TenantID.ValueString()
	}
	if !config.Region.IsNull() {
		region = config.Region.ValueString()
	}

	// Validate configuration
	if endpoint == "" {
		endpoint = "https://api.fhe-gbdt.dev"
	}

	if apiKey == "" {
		resp.Diagnostics.AddAttributeError(
			path.Root("api_key"),
			"Missing API Key",
			"The provider cannot create the FHE-GBDT API client as there is a missing or empty value for the API key. "+
				"Set the api_key value in the configuration or use the FHE_GBDT_API_KEY environment variable.",
		)
	}

	if tenantID == "" {
		resp.Diagnostics.AddAttributeError(
			path.Root("tenant_id"),
			"Missing Tenant ID",
			"The provider cannot create the FHE-GBDT API client as there is a missing or empty value for the tenant ID. "+
				"Set the tenant_id value in the configuration or use the FHE_GBDT_TENANT_ID environment variable.",
		)
	}

	if resp.Diagnostics.HasError() {
		return
	}

	// Create client
	client := NewClient(endpoint, apiKey, tenantID, region)

	resp.DataSourceData = client
	resp.ResourceData = client
}

func (p *FHEGBDTProvider) Resources(ctx context.Context) []func() resource.Resource {
	return []func() resource.Resource{
		NewModelResource,
		NewKeyResource,
		NewWebhookResource,
		NewAlertResource,
		NewTeamResource,
		NewTeamMemberResource,
	}
}

func (p *FHEGBDTProvider) DataSources(ctx context.Context) []func() datasource.DataSource {
	return []func() datasource.DataSource{
		NewModelDataSource,
		NewModelsDataSource,
		NewRegionsDataSource,
		NewUsageDataSource,
	}
}
