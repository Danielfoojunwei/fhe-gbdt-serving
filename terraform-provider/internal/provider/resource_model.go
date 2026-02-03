// Model Resource for Terraform Provider

package provider

import (
	"context"
	"fmt"

	"github.com/hashicorp/terraform-plugin-framework/path"
	"github.com/hashicorp/terraform-plugin-framework/resource"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema/planmodifier"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema/stringplanmodifier"
	"github.com/hashicorp/terraform-plugin-framework/types"
)

// Ensure provider defined types fully satisfy framework interfaces.
var _ resource.Resource = &ModelResource{}
var _ resource.ResourceWithImportState = &ModelResource{}

func NewModelResource() resource.Resource {
	return &ModelResource{}
}

// ModelResource defines the resource implementation.
type ModelResource struct {
	client *Client
}

// ModelResourceModel describes the resource data model.
type ModelResourceModel struct {
	ID             types.String `tfsdk:"id"`
	Name           types.String `tfsdk:"name"`
	Description    types.String `tfsdk:"description"`
	LibraryType    types.String `tfsdk:"library_type"`
	ModelPath      types.String `tfsdk:"model_path"`
	CompileProfile types.String `tfsdk:"compile_profile"`
	Status         types.String `tfsdk:"status"`
	CurrentVersion types.String `tfsdk:"current_version"`
	Regions        types.List   `tfsdk:"regions"`
	Labels         types.Map    `tfsdk:"labels"`
	CreatedAt      types.String `tfsdk:"created_at"`
	UpdatedAt      types.String `tfsdk:"updated_at"`
}

func (r *ModelResource) Metadata(ctx context.Context, req resource.MetadataRequest, resp *resource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_model"
}

func (r *ModelResource) Schema(ctx context.Context, req resource.SchemaRequest, resp *resource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Manages an FHE-compiled GBDT model.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Unique identifier of the model.",
				Computed:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.UseStateForUnknown(),
				},
			},
			"name": schema.StringAttribute{
				Description: "Name of the model.",
				Required:    true,
			},
			"description": schema.StringAttribute{
				Description: "Description of the model.",
				Optional:    true,
			},
			"library_type": schema.StringAttribute{
				Description: "GBDT library type: xgboost, lightgbm, or catboost.",
				Required:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.RequiresReplace(),
				},
			},
			"model_path": schema.StringAttribute{
				Description: "Path to the model file (local or S3/GCS URL).",
				Required:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.RequiresReplace(),
				},
			},
			"compile_profile": schema.StringAttribute{
				Description: "Compilation profile: fast, balanced, or accurate.",
				Optional:    true,
			},
			"status": schema.StringAttribute{
				Description: "Current status of the model.",
				Computed:    true,
			},
			"current_version": schema.StringAttribute{
				Description: "Current deployed version.",
				Computed:    true,
			},
			"regions": schema.ListAttribute{
				Description: "Regions where the model is deployed.",
				Optional:    true,
				ElementType: types.StringType,
			},
			"labels": schema.MapAttribute{
				Description: "Labels for the model.",
				Optional:    true,
				ElementType: types.StringType,
			},
			"created_at": schema.StringAttribute{
				Description: "Creation timestamp.",
				Computed:    true,
			},
			"updated_at": schema.StringAttribute{
				Description: "Last update timestamp.",
				Computed:    true,
			},
		},
	}
}

func (r *ModelResource) Configure(ctx context.Context, req resource.ConfigureRequest, resp *resource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}

	client, ok := req.ProviderData.(*Client)
	if !ok {
		resp.Diagnostics.AddError(
			"Unexpected Resource Configure Type",
			fmt.Sprintf("Expected *Client, got: %T", req.ProviderData),
		)
		return
	}

	r.client = client
}

func (r *ModelResource) Create(ctx context.Context, req resource.CreateRequest, resp *resource.CreateResponse) {
	var data ModelResourceModel

	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	// Convert regions
	var regions []string
	if !data.Regions.IsNull() {
		resp.Diagnostics.Append(data.Regions.ElementsAs(ctx, &regions, false)...)
	}

	// Convert labels
	labels := make(map[string]string)
	if !data.Labels.IsNull() {
		resp.Diagnostics.Append(data.Labels.ElementsAs(ctx, &labels, false)...)
	}

	if resp.Diagnostics.HasError() {
		return
	}

	model, err := r.client.CreateModel(ctx, &Model{
		Name:        data.Name.ValueString(),
		Description: data.Description.ValueString(),
		LibraryType: data.LibraryType.ValueString(),
		Regions:     regions,
		Labels:      labels,
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to create model", err.Error())
		return
	}

	data.ID = types.StringValue(model.ID)
	data.Status = types.StringValue(model.Status)
	data.CurrentVersion = types.StringValue(model.CurrentVersion)
	data.CreatedAt = types.StringValue(model.CreatedAt)
	data.UpdatedAt = types.StringValue(model.UpdatedAt)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *ModelResource) Read(ctx context.Context, req resource.ReadRequest, resp *resource.ReadResponse) {
	var data ModelResourceModel

	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	model, err := r.client.GetModel(ctx, data.ID.ValueString())
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
	data.UpdatedAt = types.StringValue(model.UpdatedAt)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *ModelResource) Update(ctx context.Context, req resource.UpdateRequest, resp *resource.UpdateResponse) {
	var data ModelResourceModel

	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	// Convert regions
	var regions []string
	if !data.Regions.IsNull() {
		resp.Diagnostics.Append(data.Regions.ElementsAs(ctx, &regions, false)...)
	}

	// Convert labels
	labels := make(map[string]string)
	if !data.Labels.IsNull() {
		resp.Diagnostics.Append(data.Labels.ElementsAs(ctx, &labels, false)...)
	}

	if resp.Diagnostics.HasError() {
		return
	}

	model, err := r.client.UpdateModel(ctx, data.ID.ValueString(), &Model{
		Name:        data.Name.ValueString(),
		Description: data.Description.ValueString(),
		Regions:     regions,
		Labels:      labels,
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to update model", err.Error())
		return
	}

	data.Status = types.StringValue(model.Status)
	data.UpdatedAt = types.StringValue(model.UpdatedAt)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *ModelResource) Delete(ctx context.Context, req resource.DeleteRequest, resp *resource.DeleteResponse) {
	var data ModelResourceModel

	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	err := r.client.DeleteModel(ctx, data.ID.ValueString())
	if err != nil {
		resp.Diagnostics.AddError("Failed to delete model", err.Error())
		return
	}
}

func (r *ModelResource) ImportState(ctx context.Context, req resource.ImportStateRequest, resp *resource.ImportStateResponse) {
	resource.ImportStatePassthroughID(ctx, path.Root("id"), req, resp)
}
