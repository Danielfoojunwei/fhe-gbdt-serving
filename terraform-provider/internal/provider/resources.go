// Additional Resources for Terraform Provider

package provider

import (
	"context"
	"fmt"

	"github.com/hashicorp/terraform-plugin-framework/path"
	"github.com/hashicorp/terraform-plugin-framework/resource"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema/booldefault"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema/planmodifier"
	"github.com/hashicorp/terraform-plugin-framework/resource/schema/stringplanmodifier"
	"github.com/hashicorp/terraform-plugin-framework/types"
)

// ============================================================================
// Key Resource
// ============================================================================

var _ resource.Resource = &KeyResource{}
var _ resource.ResourceWithImportState = &KeyResource{}

func NewKeyResource() resource.Resource {
	return &KeyResource{}
}

type KeyResource struct {
	client *Client
}

type KeyResourceModel struct {
	ID        types.String `tfsdk:"id"`
	Name      types.String `tfsdk:"name"`
	KeyType   types.String `tfsdk:"key_type"`
	Algorithm types.String `tfsdk:"algorithm"`
	Status    types.String `tfsdk:"status"`
	ExpiresAt types.String `tfsdk:"expires_at"`
	CreatedAt types.String `tfsdk:"created_at"`
}

func (r *KeyResource) Metadata(ctx context.Context, req resource.MetadataRequest, resp *resource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_key"
}

func (r *KeyResource) Schema(ctx context.Context, req resource.SchemaRequest, resp *resource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Manages FHE encryption keys.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Unique identifier of the key.",
				Computed:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.UseStateForUnknown(),
				},
			},
			"name": schema.StringAttribute{
				Description: "Name of the key.",
				Required:    true,
			},
			"key_type": schema.StringAttribute{
				Description: "Key type: public, private, or evaluation.",
				Required:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.RequiresReplace(),
				},
			},
			"algorithm": schema.StringAttribute{
				Description: "FHE algorithm: tfhe, ckks, or bfv.",
				Optional:    true,
			},
			"status": schema.StringAttribute{
				Description: "Current status of the key.",
				Computed:    true,
			},
			"expires_at": schema.StringAttribute{
				Description: "Expiration timestamp.",
				Optional:    true,
			},
			"created_at": schema.StringAttribute{
				Description: "Creation timestamp.",
				Computed:    true,
			},
		},
	}
}

func (r *KeyResource) Configure(ctx context.Context, req resource.ConfigureRequest, resp *resource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	r.client = req.ProviderData.(*Client)
}

func (r *KeyResource) Create(ctx context.Context, req resource.CreateRequest, resp *resource.CreateResponse) {
	var data KeyResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	key, err := r.client.CreateKey(ctx, &Key{
		Name:      data.Name.ValueString(),
		KeyType:   data.KeyType.ValueString(),
		Algorithm: data.Algorithm.ValueString(),
		ExpiresAt: data.ExpiresAt.ValueString(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to create key", err.Error())
		return
	}

	data.ID = types.StringValue(key.ID)
	data.Status = types.StringValue(key.Status)
	data.CreatedAt = types.StringValue(key.CreatedAt)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *KeyResource) Read(ctx context.Context, req resource.ReadRequest, resp *resource.ReadResponse) {
	var data KeyResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	key, err := r.client.GetKey(ctx, data.ID.ValueString())
	if err != nil {
		resp.Diagnostics.AddError("Failed to read key", err.Error())
		return
	}

	data.Name = types.StringValue(key.Name)
	data.KeyType = types.StringValue(key.KeyType)
	data.Algorithm = types.StringValue(key.Algorithm)
	data.Status = types.StringValue(key.Status)
	data.CreatedAt = types.StringValue(key.CreatedAt)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *KeyResource) Update(ctx context.Context, req resource.UpdateRequest, resp *resource.UpdateResponse) {
	resp.Diagnostics.AddError("Keys cannot be updated", "Delete and recreate the key instead.")
}

func (r *KeyResource) Delete(ctx context.Context, req resource.DeleteRequest, resp *resource.DeleteResponse) {
	var data KeyResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	if err := r.client.DeleteKey(ctx, data.ID.ValueString()); err != nil {
		resp.Diagnostics.AddError("Failed to delete key", err.Error())
	}
}

func (r *KeyResource) ImportState(ctx context.Context, req resource.ImportStateRequest, resp *resource.ImportStateResponse) {
	resource.ImportStatePassthroughID(ctx, path.Root("id"), req, resp)
}

// ============================================================================
// Webhook Resource
// ============================================================================

var _ resource.Resource = &WebhookResource{}
var _ resource.ResourceWithImportState = &WebhookResource{}

func NewWebhookResource() resource.Resource {
	return &WebhookResource{}
}

type WebhookResource struct {
	client *Client
}

type WebhookResourceModel struct {
	ID      types.String `tfsdk:"id"`
	Name    types.String `tfsdk:"name"`
	URL     types.String `tfsdk:"url"`
	Secret  types.String `tfsdk:"secret"`
	Events  types.List   `tfsdk:"events"`
	Headers types.Map    `tfsdk:"headers"`
	Enabled types.Bool   `tfsdk:"enabled"`
}

func (r *WebhookResource) Metadata(ctx context.Context, req resource.MetadataRequest, resp *resource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_webhook"
}

func (r *WebhookResource) Schema(ctx context.Context, req resource.SchemaRequest, resp *resource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Manages webhook endpoints for event notifications.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Unique identifier of the webhook.",
				Computed:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.UseStateForUnknown(),
				},
			},
			"name": schema.StringAttribute{
				Description: "Name of the webhook.",
				Required:    true,
			},
			"url": schema.StringAttribute{
				Description: "URL to receive webhook events.",
				Required:    true,
			},
			"secret": schema.StringAttribute{
				Description: "Signing secret for webhook verification.",
				Computed:    true,
				Sensitive:   true,
			},
			"events": schema.ListAttribute{
				Description: "Event types to subscribe to.",
				Required:    true,
				ElementType: types.StringType,
			},
			"headers": schema.MapAttribute{
				Description: "Custom headers to send with webhook requests.",
				Optional:    true,
				ElementType: types.StringType,
			},
			"enabled": schema.BoolAttribute{
				Description: "Whether the webhook is enabled.",
				Optional:    true,
				Computed:    true,
				Default:     booldefault.StaticBool(true),
			},
		},
	}
}

func (r *WebhookResource) Configure(ctx context.Context, req resource.ConfigureRequest, resp *resource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	r.client = req.ProviderData.(*Client)
}

func (r *WebhookResource) Create(ctx context.Context, req resource.CreateRequest, resp *resource.CreateResponse) {
	var data WebhookResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	var events []string
	resp.Diagnostics.Append(data.Events.ElementsAs(ctx, &events, false)...)

	headers := make(map[string]string)
	if !data.Headers.IsNull() {
		resp.Diagnostics.Append(data.Headers.ElementsAs(ctx, &headers, false)...)
	}

	if resp.Diagnostics.HasError() {
		return
	}

	webhook, err := r.client.CreateWebhook(ctx, &Webhook{
		Name:    data.Name.ValueString(),
		URL:     data.URL.ValueString(),
		Events:  events,
		Headers: headers,
		Enabled: data.Enabled.ValueBool(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to create webhook", err.Error())
		return
	}

	data.ID = types.StringValue(webhook.ID)
	data.Secret = types.StringValue(webhook.Secret)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *WebhookResource) Read(ctx context.Context, req resource.ReadRequest, resp *resource.ReadResponse) {
	var data WebhookResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	webhook, err := r.client.GetWebhook(ctx, data.ID.ValueString())
	if err != nil {
		resp.Diagnostics.AddError("Failed to read webhook", err.Error())
		return
	}

	data.Name = types.StringValue(webhook.Name)
	data.URL = types.StringValue(webhook.URL)
	data.Secret = types.StringValue(webhook.Secret)
	data.Enabled = types.BoolValue(webhook.Enabled)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *WebhookResource) Update(ctx context.Context, req resource.UpdateRequest, resp *resource.UpdateResponse) {
	var data WebhookResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	var events []string
	resp.Diagnostics.Append(data.Events.ElementsAs(ctx, &events, false)...)

	headers := make(map[string]string)
	if !data.Headers.IsNull() {
		resp.Diagnostics.Append(data.Headers.ElementsAs(ctx, &headers, false)...)
	}

	if resp.Diagnostics.HasError() {
		return
	}

	webhook, err := r.client.UpdateWebhook(ctx, data.ID.ValueString(), &Webhook{
		Name:    data.Name.ValueString(),
		URL:     data.URL.ValueString(),
		Events:  events,
		Headers: headers,
		Enabled: data.Enabled.ValueBool(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to update webhook", err.Error())
		return
	}

	data.Secret = types.StringValue(webhook.Secret)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *WebhookResource) Delete(ctx context.Context, req resource.DeleteRequest, resp *resource.DeleteResponse) {
	var data WebhookResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	if err := r.client.DeleteWebhook(ctx, data.ID.ValueString()); err != nil {
		resp.Diagnostics.AddError("Failed to delete webhook", err.Error())
	}
}

func (r *WebhookResource) ImportState(ctx context.Context, req resource.ImportStateRequest, resp *resource.ImportStateResponse) {
	resource.ImportStatePassthroughID(ctx, path.Root("id"), req, resp)
}

// ============================================================================
// Alert Resource
// ============================================================================

var _ resource.Resource = &AlertResource{}
var _ resource.ResourceWithImportState = &AlertResource{}

func NewAlertResource() resource.Resource {
	return &AlertResource{}
}

type AlertResource struct {
	client *Client
}

type AlertResourceModel struct {
	ID            types.String  `tfsdk:"id"`
	Name          types.String  `tfsdk:"name"`
	ModelID       types.String  `tfsdk:"model_id"`
	Metric        types.String  `tfsdk:"metric"`
	Condition     types.String  `tfsdk:"condition"`
	Threshold     types.Float64 `tfsdk:"threshold"`
	WindowMinutes types.Int64   `tfsdk:"window_minutes"`
	Channels      types.List    `tfsdk:"notification_channels"`
	Enabled       types.Bool    `tfsdk:"enabled"`
}

func (r *AlertResource) Metadata(ctx context.Context, req resource.MetadataRequest, resp *resource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_alert"
}

func (r *AlertResource) Schema(ctx context.Context, req resource.SchemaRequest, resp *resource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Manages monitoring alerts for models.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Unique identifier of the alert.",
				Computed:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.UseStateForUnknown(),
				},
			},
			"name": schema.StringAttribute{
				Description: "Name of the alert.",
				Required:    true,
			},
			"model_id": schema.StringAttribute{
				Description: "Model ID to monitor.",
				Required:    true,
			},
			"metric": schema.StringAttribute{
				Description: "Metric to monitor: latency_p95, error_rate, throughput, drift.",
				Required:    true,
			},
			"condition": schema.StringAttribute{
				Description: "Condition: gt, gte, lt, lte, eq.",
				Required:    true,
			},
			"threshold": schema.Float64Attribute{
				Description: "Threshold value.",
				Required:    true,
			},
			"window_minutes": schema.Int64Attribute{
				Description: "Evaluation window in minutes.",
				Required:    true,
			},
			"notification_channels": schema.ListAttribute{
				Description: "Notification channels: email, slack, pagerduty, webhook.",
				Required:    true,
				ElementType: types.StringType,
			},
			"enabled": schema.BoolAttribute{
				Description: "Whether the alert is enabled.",
				Optional:    true,
				Computed:    true,
				Default:     booldefault.StaticBool(true),
			},
		},
	}
}

func (r *AlertResource) Configure(ctx context.Context, req resource.ConfigureRequest, resp *resource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	r.client = req.ProviderData.(*Client)
}

func (r *AlertResource) Create(ctx context.Context, req resource.CreateRequest, resp *resource.CreateResponse) {
	var data AlertResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	var channels []string
	resp.Diagnostics.Append(data.Channels.ElementsAs(ctx, &channels, false)...)
	if resp.Diagnostics.HasError() {
		return
	}

	alert, err := r.client.CreateAlert(ctx, &Alert{
		Name:          data.Name.ValueString(),
		ModelID:       data.ModelID.ValueString(),
		Metric:        data.Metric.ValueString(),
		Condition:     data.Condition.ValueString(),
		Threshold:     data.Threshold.ValueFloat64(),
		WindowMinutes: int(data.WindowMinutes.ValueInt64()),
		Channels:      channels,
		Enabled:       data.Enabled.ValueBool(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to create alert", err.Error())
		return
	}

	data.ID = types.StringValue(alert.ID)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *AlertResource) Read(ctx context.Context, req resource.ReadRequest, resp *resource.ReadResponse) {
	var data AlertResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	alert, err := r.client.GetAlert(ctx, data.ID.ValueString())
	if err != nil {
		resp.Diagnostics.AddError("Failed to read alert", err.Error())
		return
	}

	data.Name = types.StringValue(alert.Name)
	data.ModelID = types.StringValue(alert.ModelID)
	data.Metric = types.StringValue(alert.Metric)
	data.Condition = types.StringValue(alert.Condition)
	data.Threshold = types.Float64Value(alert.Threshold)
	data.WindowMinutes = types.Int64Value(int64(alert.WindowMinutes))
	data.Enabled = types.BoolValue(alert.Enabled)

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *AlertResource) Update(ctx context.Context, req resource.UpdateRequest, resp *resource.UpdateResponse) {
	var data AlertResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	var channels []string
	resp.Diagnostics.Append(data.Channels.ElementsAs(ctx, &channels, false)...)
	if resp.Diagnostics.HasError() {
		return
	}

	_, err := r.client.UpdateAlert(ctx, data.ID.ValueString(), &Alert{
		Name:          data.Name.ValueString(),
		ModelID:       data.ModelID.ValueString(),
		Metric:        data.Metric.ValueString(),
		Condition:     data.Condition.ValueString(),
		Threshold:     data.Threshold.ValueFloat64(),
		WindowMinutes: int(data.WindowMinutes.ValueInt64()),
		Channels:      channels,
		Enabled:       data.Enabled.ValueBool(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to update alert", err.Error())
		return
	}

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *AlertResource) Delete(ctx context.Context, req resource.DeleteRequest, resp *resource.DeleteResponse) {
	var data AlertResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	if err := r.client.DeleteAlert(ctx, data.ID.ValueString()); err != nil {
		resp.Diagnostics.AddError("Failed to delete alert", err.Error())
	}
}

func (r *AlertResource) ImportState(ctx context.Context, req resource.ImportStateRequest, resp *resource.ImportStateResponse) {
	resource.ImportStatePassthroughID(ctx, path.Root("id"), req, resp)
}

// ============================================================================
// Team Resource
// ============================================================================

var _ resource.Resource = &TeamResource{}

func NewTeamResource() resource.Resource {
	return &TeamResource{}
}

type TeamResource struct {
	client *Client
}

type TeamResourceModel struct {
	ID          types.String `tfsdk:"id"`
	Name        types.String `tfsdk:"name"`
	Description types.String `tfsdk:"description"`
}

func (r *TeamResource) Metadata(ctx context.Context, req resource.MetadataRequest, resp *resource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_team"
}

func (r *TeamResource) Schema(ctx context.Context, req resource.SchemaRequest, resp *resource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Manages teams within an organization.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Unique identifier of the team.",
				Computed:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.UseStateForUnknown(),
				},
			},
			"name": schema.StringAttribute{
				Description: "Name of the team.",
				Required:    true,
			},
			"description": schema.StringAttribute{
				Description: "Description of the team.",
				Optional:    true,
			},
		},
	}
}

func (r *TeamResource) Configure(ctx context.Context, req resource.ConfigureRequest, resp *resource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	r.client = req.ProviderData.(*Client)
}

func (r *TeamResource) Create(ctx context.Context, req resource.CreateRequest, resp *resource.CreateResponse) {
	var data TeamResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	team, err := r.client.CreateTeam(ctx, &Team{
		Name:        data.Name.ValueString(),
		Description: data.Description.ValueString(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to create team", err.Error())
		return
	}

	data.ID = types.StringValue(team.ID)
	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *TeamResource) Read(ctx context.Context, req resource.ReadRequest, resp *resource.ReadResponse) {
	var data TeamResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	team, err := r.client.GetTeam(ctx, data.ID.ValueString())
	if err != nil {
		resp.Diagnostics.AddError("Failed to read team", err.Error())
		return
	}

	data.Name = types.StringValue(team.Name)
	data.Description = types.StringValue(team.Description)
	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *TeamResource) Update(ctx context.Context, req resource.UpdateRequest, resp *resource.UpdateResponse) {
	var data TeamResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	_, err := r.client.UpdateTeam(ctx, data.ID.ValueString(), &Team{
		Name:        data.Name.ValueString(),
		Description: data.Description.ValueString(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to update team", err.Error())
		return
	}

	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *TeamResource) Delete(ctx context.Context, req resource.DeleteRequest, resp *resource.DeleteResponse) {
	var data TeamResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	if err := r.client.DeleteTeam(ctx, data.ID.ValueString()); err != nil {
		resp.Diagnostics.AddError("Failed to delete team", err.Error())
	}
}

// ============================================================================
// Team Member Resource
// ============================================================================

var _ resource.Resource = &TeamMemberResource{}

func NewTeamMemberResource() resource.Resource {
	return &TeamMemberResource{}
}

type TeamMemberResource struct {
	client *Client
}

type TeamMemberResourceModel struct {
	ID     types.String `tfsdk:"id"`
	TeamID types.String `tfsdk:"team_id"`
	Email  types.String `tfsdk:"email"`
	Role   types.String `tfsdk:"role"`
}

func (r *TeamMemberResource) Metadata(ctx context.Context, req resource.MetadataRequest, resp *resource.MetadataResponse) {
	resp.TypeName = req.ProviderTypeName + "_team_member"
}

func (r *TeamMemberResource) Schema(ctx context.Context, req resource.SchemaRequest, resp *resource.SchemaResponse) {
	resp.Schema = schema.Schema{
		Description: "Manages team membership.",
		Attributes: map[string]schema.Attribute{
			"id": schema.StringAttribute{
				Description: "Unique identifier of the membership.",
				Computed:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.UseStateForUnknown(),
				},
			},
			"team_id": schema.StringAttribute{
				Description: "Team ID.",
				Required:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.RequiresReplace(),
				},
			},
			"email": schema.StringAttribute{
				Description: "Email of the member to add.",
				Required:    true,
				PlanModifiers: []planmodifier.String{
					stringplanmodifier.RequiresReplace(),
				},
			},
			"role": schema.StringAttribute{
				Description: "Role: admin, member, viewer.",
				Required:    true,
			},
		},
	}
}

func (r *TeamMemberResource) Configure(ctx context.Context, req resource.ConfigureRequest, resp *resource.ConfigureResponse) {
	if req.ProviderData == nil {
		return
	}
	r.client = req.ProviderData.(*Client)
}

func (r *TeamMemberResource) Create(ctx context.Context, req resource.CreateRequest, resp *resource.CreateResponse) {
	var data TeamMemberResourceModel
	resp.Diagnostics.Append(req.Plan.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	member, err := r.client.AddTeamMember(ctx, &TeamMember{
		TeamID: data.TeamID.ValueString(),
		Email:  data.Email.ValueString(),
		Role:   data.Role.ValueString(),
	})
	if err != nil {
		resp.Diagnostics.AddError("Failed to add team member", err.Error())
		return
	}

	data.ID = types.StringValue(member.ID)
	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *TeamMemberResource) Read(ctx context.Context, req resource.ReadRequest, resp *resource.ReadResponse) {
	var data TeamMemberResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	// Read implementation would fetch member details
	resp.Diagnostics.Append(resp.State.Set(ctx, &data)...)
}

func (r *TeamMemberResource) Update(ctx context.Context, req resource.UpdateRequest, resp *resource.UpdateResponse) {
	resp.Diagnostics.AddError("Team members cannot be updated", "Remove and re-add with new role.")
}

func (r *TeamMemberResource) Delete(ctx context.Context, req resource.DeleteRequest, resp *resource.DeleteResponse) {
	var data TeamMemberResourceModel
	resp.Diagnostics.Append(req.State.Get(ctx, &data)...)
	if resp.Diagnostics.HasError() {
		return
	}

	if err := r.client.RemoveTeamMember(ctx, data.TeamID.ValueString(), data.ID.ValueString()); err != nil {
		resp.Diagnostics.AddError("Failed to remove team member", err.Error())
	}
}
