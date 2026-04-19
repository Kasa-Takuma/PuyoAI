export const SEARCH_PROFILES = Object.freeze([
  {
    id: "chain_builder_v3",
    label: "Chain Builder v3",
    description: "Current default profile focused on larger chains and strong virtual fires.",
  },
  {
    id: "chain_builder_v4",
    label: "Chain Builder v4",
    description: "Experimental profile that rewards 10+ chain potential while guarding high stacks without firepower.",
  },
  {
    id: "chain_builder_v5",
    label: "Chain Builder v5",
    description: "Pushes v4 toward 10+ finishes by favoring stronger virtual firepower over early 7-9 chain cashouts.",
  },
  {
    id: "chain_builder_v6",
    label: "Chain Builder v6",
    description: "Keeps v4's high-chain frequency while adding a lighter 10+ preference than v5.",
  },
  {
    id: "chain_builder_v7",
    label: "Chain Builder v7",
    description: "Stretches v6 toward cleaner 11+ and 12-chain potential without dropping 10-chain pressure too hard.",
  },
  {
    id: "chain_builder_v7a",
    label: "Chain Builder v7a",
    description: "Bolder stable-frequency profile that suppresses 7-9 early fires and strongly favors repeatable 10+ chain-ready boards.",
  },
  {
    id: "chain_builder_v8",
    label: "Chain Builder v8",
    description: "Builds on v7a with stronger anti-small-fire penalties while preserving cleanup moves that lead to 10+ potential.",
  },
  {
    id: "chain_builder_v9",
    label: "Chain Builder v9",
    description: "Pushes v8 toward steadier 11-12 chains by making 7-9 fires less attractive and rewarding mature 11+ boards.",
  },
  {
    id: "chain_builder_v9a",
    label: "Chain Builder v9a",
    description: "Focus-tuned 11+ profile that rewards the mature high-stack patterns seen before v8's 11-12 chains.",
  },
  {
    id: "chain_builder_v9b",
    label: "Chain Builder v9b",
    description: "Keeps v9's flexibility while lightly biasing choices toward 11+ over 10-chain finishes.",
  },
]);

export const DEFAULT_SEARCH_PROFILE_ID = SEARCH_PROFILES[0].id;

export function getSearchProfile(profileId) {
  return (
    SEARCH_PROFILES.find((profile) => profile.id === profileId) ??
    SEARCH_PROFILES[0]
  );
}
