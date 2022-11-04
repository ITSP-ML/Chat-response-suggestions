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

  function format_label(suggestion) {
    return suggestion.prefix + suggestion.sugg
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
  labelFunction={format_label}
  maxItemsToShowInList={50}
  delay={200}
  localFiltering={false}
  cleanUserText={false}
  hideArrow={true}
  minCharactersToSearch={0}
  style="width:1000px;">
    <div slot="item" let:item let:label>
      {@html item.sugg.replace(' ', '<span style="color:lightblue">_</span>')}
      <!-- to render the default highlighted item label -->
      <!-- render anything else -->
      <i style="color:grey">({(100*item.prob).toFixed(2)}%)</i>
    </div>
    <div slot="loading" let:loadingText>
        <i style="color:grey">searching for suggestions</i>
    </div>
</AutoComplete>
