"""Minimal Stash GraphQL client plus a cached tag manager."""
import requests

FIND_IDS = {
    "image": ("query($f:FindFilterType,$x:ImageFilterType)"
              "{findImages(filter:$f,image_filter:$x){images{id}}}", "findImages", "images"),
    "scene": ("query($f:FindFilterType,$x:SceneFilterType)"
              "{findScenes(filter:$f,scene_filter:$x){scenes{id}}}", "findScenes", "scenes"),
}
BULK_UPDATE = {
    "image": "mutation($i:BulkImageUpdateInput!){bulkImageUpdate(input:$i){id}}",
    "scene": "mutation($i:BulkSceneUpdateInput!){bulkSceneUpdate(input:$i){id}}",
}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def tag_filter(tag_id, modifier="INCLUDES"):
    return {"tags": {"value": [tag_id], "modifier": modifier, "depth": 0}}


class Stash:
    def __init__(self, url, api_key=""):
        self.url = url
        self.session = requests.Session()
        if api_key:
            self.session.headers["ApiKey"] = api_key

    def gql(self, query, variables=None):
        r = self.session.post(self.url, json={"query": query, "variables": variables or {}},
                              timeout=300)
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            raise RuntimeError(data["errors"])
        return data["data"]

    def get_bytes(self, url):
        r = self.session.get(url, timeout=300)
        r.raise_for_status()
        return r.content

    # --- items ---------------------------------------------------------------
    def find_ids(self, kind, item_filter=None):
        query, root, key = FIND_IDS[kind]
        data = self.gql(query, {"f": {"per_page": -1}, "x": item_filter or {}})
        return [x["id"] for x in data[root][key]]

    def image_info(self, image_id):
        q = "query($id:ID!){findImage(id:$id){id paths{image}}}"
        return self.gql(q, {"id": image_id})["findImage"]

    def scene_info(self, scene_id):
        q = "query($id:ID!){findScene(id:$id){id files{path duration}}}"
        return self.gql(q, {"id": scene_id})["findScene"]

    def update_tags(self, kind, ids, tag_ids, mode="ADD"):
        """mode: ADD keeps existing tags, REMOVE removes only these tags."""
        self.gql(BULK_UPDATE[kind], {"i": {"ids": ids, "tag_ids": {"ids": tag_ids, "mode": mode}}})

    # --- tags ------------------------------------------------------------------
    def all_tags(self):
        q = "query{findTags(filter:{per_page:-1}){tags{id name aliases}}}"
        return self.gql(q)["findTags"]["tags"]

    def create_tag(self, name, aliases=None, parent_ids=None, description=None):
        q = "mutation($i:TagCreateInput!){tagCreate(input:$i){id}}"
        inp = {"name": name, "ignore_auto_tag": True}  # keep Stash's filename auto-tagger off these
        if aliases:
            inp["aliases"] = aliases
        if parent_ids:
            inp["parent_ids"] = parent_ids
        if description:
            inp["description"] = description
        return self.gql(q, {"i": inp})["tagCreate"]["id"]


class TagManager:
    """Looks tags up by name or alias (case-insensitive); creates missing ones once."""

    def __init__(self, stash, dry_run=False):
        self.stash = stash
        self.dry_run = dry_run
        self.by_key = {}
        self._fake = 0
        for t in stash.all_tags():
            self.by_key[t["name"].lower()] = t["id"]
            for a in t.get("aliases") or []:
                self.by_key.setdefault(a.lower(), t["id"])

    def find(self, name):
        return self.by_key.get(name.lower())

    def get(self, name, aliases=(), parents=(), description=None):
        aliases = [a for a in aliases if a.lower() != name.lower()]
        for key in [name, *aliases]:
            tid = self.by_key.get(key.lower())
            if tid:
                return tid
        parent_ids = [p for p in parents if p and not str(p).startswith("dry-")]
        if self.dry_run:
            self._fake += 1
            tid = f"dry-{self._fake}"
            print(f"  [dry-run] would create tag {name!r}")
        else:
            try:
                tid = self.stash.create_tag(name, aliases, parent_ids, description)
            except RuntimeError:
                if not aliases:
                    raise
                # alias clashes with another tag's name/alias: create without it
                tid = self.stash.create_tag(name, [], parent_ids, description)
        self.by_key[name.lower()] = tid
        for a in aliases:
            self.by_key.setdefault(a.lower(), tid)
        return tid
