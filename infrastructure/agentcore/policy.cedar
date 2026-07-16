// Cedar policy for the ATX NKI agent.
// Filesystem access is scoped to /tmp/nki-agent, network egress is gated to
// the explicit allowlist in agent_config.yaml.

permit (
    principal,
    action == Action::"file::read",
    resource
) when {
    resource.path like "/tmp/nki-agent/*"
};

permit (
    principal,
    action == Action::"file::write",
    resource
) when {
    resource.path like "/tmp/nki-agent/*"
};

forbid (
    principal,
    action == Action::"file::write",
    resource
) when {
    not (resource.path like "/tmp/nki-agent/*")
};

permit (
    principal,
    action == Action::"network::egress",
    resource
) when {
    resource.host in ["bedrock.us-east-1.amazonaws.com",
                      "bedrock-runtime.us-east-1.amazonaws.com"]
};
