// Data Sources for Terraform Provider

package provider

import (
	"context"

	"github.com/hashicorp/terraform-plugin-framework/datasource"
	"github.com/hashicorp/terraform-plugin-framework/datasource/schema"
	"github.com/hashicorp/terraform-plugin-framework/types"
)

// ============================================================================
// Model Data Source
// ============================================================================

var _ datasource.DataSource = &ModelDataSource{}

func NewModelDataSource() datasource.DataSource {
	return &ModelDataSource{}
}

type ModelDataSource struct {
	client *Client
}

type ModelDataSourceModel struct {
	ID             types.String `tfsdk:"id"`
	Name           types.String `tfsdk:"name"`
	Description    types.String `tfsdk:"description"`
	LibraryType    types.String `tfsdk:"library_type"`
	Status         types.String `tfsdk:"status"`
	CurrentVersion types.String `tfsdk:"current_version"`
	CreatedAt      types.String `tfsdk:"created_at"`
}

func (d *ModelDataSource) Metadata(ctx context.Context, req datasource.MetadataRequest, resp *datasource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_model"
}

func (d *ModelDataSource) Schema(ctx context.Context, req datasource.SchemaRequest, resp *datasource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Fetches information about an FHE-GBDT model.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Model ID to look up.",
				Required:    true,
			},
			"name": schema.StringAttribute{
				Description: "Name of the model.",
				Computed:    true,
			},
			"description": schema.StringAttribute{
				Description: "Description of the model.",
				Computed:    true,
			},
			"library_type": schema.StringAttribute{
				Description: "GBDT library type.",
				Computed:    true,
			},
			"status": schema.StringAttribute{
				Description: "Current status.",
				Computed:    true,
			},
			"current_version": schema.StringAttribute{
				Description: "Current deployed version.",
				Computed:    true,
			},
			"created_at": schema.StringAttribute{
				Description: "Creation timestamp.",
				Computed:    true,
			},
		},
	}
}

func (d *ModelDataSource) Configure(ctx context.Context, req datasource.ConfigureRequest, resp *datasource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	d.client = req.ProviderData.(*Client)
}

func (d *ModelDataSource) Read(ctx context.Context, req datasource.ReadRequest, resp *datasource.ReadResponse) {
	var data ModelDataSourceModel
	resp.Diagnostics.Append(req.Config.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	model, err := d.client.GetModel(ctx, data.ID.ValueString())
	if err != nil {
		resp.Diagnostics.AddError("Failed to read model", err.Error())
		return
	}

	data.Name = types.StringValue(model.Name)
	data.Description = types.StringValue(model.Description)
	data.LibraryType = types.StringValue(model.LibraryType)
	data.Status = types.StringValue(model.Status)
	data.CurrentVersion = types.StringValue(model.CurrentVersion)
	data.CreatedAt = types.StringValue(model.CreatedAt)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

// ============================================================================
// Models Data Source (List)
// ============================================================================

var _ datasource.DataSource = &ModelsDataSource{}

func NewModelsDataSource() datasource.DataSource {
	return &ModelsDataSource{}
}

type ModelsDataSource struct {
	client *Client
}

type ModelsDataSourceModel struct {
	Models []ModelDataSourceModel `tfsdk:"models"`
}

func (d *ModelsDataSource) Metadata(ctx context.Context, req datasource.MetadataRequest, resp *datasource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_models"
}

func (d *ModelsDataSource) Schema(ctx context.Context, req datasource.SchemaRequest, resp *datasource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Lists all FHE-GBDT models.",
		Attributes: map[string]schema.Attribute{
			"models": schema.ListNestedAttribute{
				Description: "List of models.",
				Computed:    true,
				NestedObject: schema.NestedAttributeObject{
					Attributes: map[string]schema.Attribute{
						"id": schema.StringAttribute{
							Computed: true,
						},
						"name": schema.StringAttribute{
							Computed: true,
						},
						"description": schema.StringAttribute{
							Computed: true,
						},
						"library_type": schema.StringAttribute{
							Computed: true,
						},
						"status": schema.StringAttribute{
							Computed: true,
						},
						"current_version": schema.StringAttribute{
							Computed: true,
						},
						"created_at": schema.StringAttribute{
							Computed: true,
						},
					},
				},
			},
		},
	}
}

func (d *ModelsDataSource) Configure(ctx context.Context, req datasource.ConfigureRequest, resp *datasource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	d.client = req.ProviderData.(*Client)
}

func (d *ModelsDataSource) Read(ctx context.Context, req datasource.ReadRequest, resp *datasource.ReadResponse) {
	var data ModelsDataSourceModel

	models, err := d.client.ListModels(ctx)
	if err != nil {
		resp.Diagnostics.AddError("Failed to list models", err.Error())
		return
	}

	for _, model := range models {
		data.Models = append(data.Models, ModelDataSourceModel{
			ID:             types.StringValue(model.ID),
			Name:           types.StringValue(model.Name),
			Description:    types.StringValue(model.Description),
			LibraryType:    types.StringValue(model.LibraryType),
			Status:         types.StringValue(model.Status),
			CurrentVersion: types.StringValue(model.CurrentVersion),
			CreatedAt:      types.StringValue(model.CreatedAt),
		})
	}

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

// ============================================================================
// Regions Data Source
// ============================================================================

var _ datasource.DataSource = &RegionsDataSource{}

func NewRegionsDataSource() datasource.DataSource {
	return &RegionsDataSource{}
}

type RegionsDataSource struct {
	client *Client
}

type RegionDataSourceModel struct {
	Code       types.String `tfsdk:"code"`
	Name       types.String `tfsdk:"name"`
	Provider   types.String `tfsdk:"provider"`
	Country    types.String `tfsdk:"country"`
	Status     types.String `tfsdk:"status"`
	Compliance types.List   `tfsdk:"compliance"`
}

type RegionsDataSourceModel struct {
	Regions []RegionDataSourceModel `tfsdk:"regions"`
}

func (d *RegionsDataSource) Metadata(ctx context.Context, req datasource.MetadataRequest, resp *datasource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_regions"
}

func (d *RegionsDataSource) Schema(ctx context.Context, req datasource.SchemaRequest, resp *datasource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Lists available deployment regions.",
		Attributes: map[string]schema.Attribute{
			"regions": schema.ListNestedAttribute{
				Description: "List of regions.",
				Computed:    true,
				NestedObject: schema.NestedAttributeObject{
					Attributes: map[string]schema.Attribute{
						"code": schema.StringAttribute{
							Description: "Region code (e.g., us-east-1).",
							Computed:    true,
						},
						"name": schema.StringAttribute{
							Description: "Region name.",
							Computed:    true,
						},
						"provider": schema.StringAttribute{
							Description: "Cloud provider.",
							Computed:    true,
						},
						"country": schema.StringAttribute{
							Description: "Country code.",
							Computed:    true,
						},
						"status": schema.StringAttribute{
							Description: "Region status.",
							Computed:    true,
						},
						"compliance": schema.ListAttribute{
							Description: "Compliance certifications.",
							Computed:    true,
							ElementType: types.StringType,
						},
					},
				},
			},
		},
	}
}

func (d *RegionsDataSource) Configure(ctx context.Context, req datasource.ConfigureRequest, resp *datasource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	d.client = req.ProviderData.(*Client)
}

func (d *RegionsDataSource) Read(ctx context.Context, req datasource.ReadRequest, resp *datasource.ReadResponse) {
	var data RegionsDataSourceModel

	regions, err := d.client.ListRegions(ctx)
	if err != nil {
		resp.Diagnostics.AddError("Failed to list regions", err.Error())
		return
	}

	for _, region := range regions {
		compliance, _ := types.ListValueFrom(ctx, types.StringType, region.Compliance)
		data.Regions = append(data.Regions, RegionDataSourceModel{
			Code:       types.StringValue(region.Code),
			Name:       types.StringValue(region.Name),
			Provider:   types.StringValue(region.Provider),
			Country:    types.StringValue(region.Country),
			Status:     types.StringValue(region.Status),
			Compliance: compliance,
		})
	}

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

// ============================================================================
// Usage Data Source
// ============================================================================

var _ datasource.DataSource = &UsageDataSource{}

func NewUsageDataSource() datasource.DataSource {
	return &UsageDataSource{}
}

type UsageDataSource struct {
	client *Client
}

type UsageDataSourceModel struct {
	TotalPredictions types.Int64   `tfsdk:"total_predictions"`
	TotalCompute     types.Float64 `tfsdk:"total_compute_hours"`
	TotalStorage     types.Float64 `tfsdk:"total_storage_gb"`
	PeriodStart      types.String  `tfsdk:"period_start"`
	PeriodEnd        types.String  `tfsdk:"period_end"`
}

func (d *UsageDataSource) Metadata(ctx context.Context, req datasource.MetadataRequest, resp *datasource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_usage"
}

func (d *UsageDataSource) Schema(ctx context.Context, req datasource.SchemaRequest, resp *datasource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Fetches current usage metrics.",
		Attributes: map[string]schema.Attribute{
			"total_predictions": schema.Int64Attribute{
				Description: "Total predictions in current period.",
				Computed:    true,
			},
			"total_compute_hours": schema.Float64Attribute{
				Description: "Total compute hours used.",
				Computed:    true,
			},
			"total_storage_gb": schema.Float64Attribute{
				Description: "Total storage used in GB.",
				Computed:    true,
			},
			"period_start": schema.StringAttribute{
				Description: "Billing period start.",
				Computed:    true,
			},
			"period_end": schema.StringAttribute{
				Description: "Billing period end.",
				Computed:    true,
			},
		},
	}
}

func (d *UsageDataSource) Configure(ctx context.Context, req datasource.ConfigureRequest, resp *datasource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	d.client = req.ProviderData.(*Client)
}

func (d *UsageDataSource) Read(ctx context.Context, req datasource.ReadRequest, resp *datasource.ReadResponse) {
	usage, err := d.client.GetUsage(ctx)
	if err != nil {
		resp.Diagnostics.AddError("Failed to get usage", err.Error())
		return
	}

	data := UsageDataSourceModel{
		TotalPredictions: types.Int64Value(usage.TotalPredictions),
		TotalCompute:     types.Float64Value(usage.TotalCompute),
		TotalStorage:     types.Float64Value(usage.TotalStorage),
		PeriodStart:      types.StringValue(usage.PeriodStart),
		PeriodEnd:        types.StringValue(usage.PeriodEnd),
	}

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}
