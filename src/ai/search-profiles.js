export const SEARCH_PROFILES = Object.freeze([
  {
    id: "chain_builder_v3",
    label: "Chain Builder v3",
    description: "Current default profile focused on larger chains and strong virtual fires.",
  },
  {
    id: "balanced_v1",
    label: "Balanced v1",
    description: "Less extreme than the default, aiming for steadier stacking and cleaner shapes.",
  },
  {
    id: "survival_v1",
    label: "Survival v1",
    description: "Prioritizes lower stacks and smoother surfaces to reduce topouts.",
  },
  {
    id: "aggressive_chain_v1",
    label: "Aggressive Chain v1",
    description: "Pushes harder toward virtual large-chain opportunities even if the stack gets riskier.",
  },
  {
    id: "long_horizon_v1",
    label: "Long Horizon v1",
    description: "Tries to preserve extendable shapes for longer before cashing in.",
  },
]);

export const DEFAULT_SEARCH_PROFILE_ID = SEARCH_PROFILES[0].id;

export function getSearchProfile(profileId) {
  return (
    SEARCH_PROFILES.find((profile) => profile.id === profileId) ??
    SEARCH_PROFILES[0]
  );
}
