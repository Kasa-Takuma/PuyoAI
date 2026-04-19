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
]);

export const DEFAULT_SEARCH_PROFILE_ID = SEARCH_PROFILES[0].id;

export function getSearchProfile(profileId) {
  return (
    SEARCH_PROFILES.find((profile) => profile.id === profileId) ??
    SEARCH_PROFILES[0]
  );
}
