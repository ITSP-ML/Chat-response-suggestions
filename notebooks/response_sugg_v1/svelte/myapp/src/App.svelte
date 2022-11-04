<script>
  import AutoComplete from "simple-svelte-autocomplete";
  let selectedSugg;

  async function get_suggestions_list(keyword) {
    let data = { text: keyword };

    const url = "http://127.0.0.1:8000/";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    return response.json();
  }

//  function onKeyDown(e) {
//    if (e.key == 'Tab') {
//
//    }
//  }
</script>

<AutoComplete
  searchFunction={get_suggestions_list}
  bind:selectedItem={selectedSugg}
  keywordsFieldName="sugg"
  labelFunction={suggestion => `${suggestion.sugg} (${(100*suggestion.prob).toFixed(2)}%)`}
  maxItemsToShowInList={10}
  delay={300}
  localFiltering={false}
  cleanUserText={false}
/>
