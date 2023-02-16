<script>
  import AutoComplete from "simple-svelte-autocomplete";
  let selectedCountry;
  async function get_suggestions_list(keyword) {
    // let x = JSON.stringify({ text: "P" });
    let data = { text: keyword };
    const url = "http://172.26.103.221:5010/autocomplete/search"; // for AutoComplete
    // const url = "http://172.26.103.221:5010/semantic_search/search"; // for semantic search
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    const json = await response.json()
    return json.suggestions.map(x => ({"sugg": x})); 
  }
</script>

<AutoComplete
  searchFunction={get_suggestions_list}
  bind:selectedItem={selectedCountry}
  labelFieldName="sugg"
  maxItemsToShowInList={10}
  delay={200}
  localFiltering={false}
  lowercaseKeywords={true}
/>
