defmodule Transmuter.Vision.Florence2Processor do
  @moduledoc """
  The prompts and the post-processing of Florence-2, as `Florence2Processor` of Python transformers 4.57. `Transmuter.Vision.image_to_text/2` uses it for a repository whose encoder takes `inputs_embeds`.

  A task is a task token, such as `"<CAPTION>"` or `"<OD>"`. The tables of the tasks come from `preprocessor_config.json`, as transformers.js reads them: `task_prompts_without_inputs` and `task_prompts_with_input` give the prompt of each task, and `tasks_answer_post_processing_type` gives the post-processing of its answer. A task with input puts the text of the input in the place of `{input}` of its prompt.

  The answer of a location task has location tokens `<loc_N>`, with `N` from 0 to 999. A location token is the center of a bin of 1/1000 of the width or the height of the image, so the coordinate is `(N + 0.5) * width / 1000` for x and `(N + 0.5) * height / 1000` for y. The coordinates are floats, in pixels of the input image. Python transformers 4.57 truncates them to integers, and its remote code of `microsoft/Florence-2-base-ft` gives floats.
  """

  alias Transmuter.Hub

  # The phrases that the phrase grounding of Python transformers ignores. They are in
  # processor_config.json of florence-community/Florence-2-base-ft and in
  # _create_black_list_of_phrase_grounding of the remote code of microsoft/Florence-2-base-ft.
  # The ONNX exports do not have them.
  @banned_grounding_tokens MapSet.new(
                             ~w(it I me mine you your yours he him his she her hers
    they them their theirs one oneself we us our ours its myself yourself himself herself itself
    ourselves yourselves themselves this that these those who whom whose which what all another
    any anybody anyone anything each everybody everyone everything few many nobody none several
    some somebody someone something image images the a an lots) ++
                               [
                                 "each other",
                                 "one another",
                                 "the image",
                                 "a group",
                                 "other objects",
                                 "a set"
                               ]
                           )

  @doc """
  Loads the tables of the tasks from `preprocessor_config.json` of `repo`. Raises when the configuration does not have them.
  """
  def load!(repo) do
    config = Hub.json(repo.source, "preprocessor_config.json", repo.opts) || %{}

    unless is_map(config["task_prompts_without_inputs"]) do
      raise ArgumentError,
            "preprocessor_config.json of #{inspect(repo.source)} has no task_prompts_without_inputs, so the tasks of Florence-2 are not known"
    end

    %{
      prompts: config["task_prompts_without_inputs"],
      input_prompts: config["task_prompts_with_input"] || %{},
      types: config["tasks_answer_post_processing_type"] || %{},
      image_tokens: config["image_seq_length"]
    }
  end

  @doc """
  Returns the prompt of `task` with the input `text`, or `nil` for no input.
  """
  def prompt!(processor, task, text) do
    cond do
      prompt = processor.prompts[task] ->
        if text != nil, do: raise(ArgumentError, "the task #{task} takes no :text")
        prompt

      prompt = processor.input_prompts[task] ->
        String.replace(prompt, "{input}", String.trim(text || ""))

      true ->
        tasks = Map.keys(processor.prompts) ++ Map.keys(processor.input_prompts)

        raise ArgumentError,
              "expected :task to be one of #{Enum.map_join(Enum.sort(tasks), ", ", &inspect/1)}, got: #{inspect(task)}"
    end
  end

  @doc """
  Returns the result of `task` from the generated `text`, which has the special tokens, for an image of size `{height, width}`.

  The result always has `:text`: the text without `<s>` and `</s>`, and without spaces at the ends. The post-processing type of the task adds keys:

    * `"pure_text"` - no other keys.

    * `"description_with_bboxes"`, `"bboxes"` and `"phrase_grounding"` - `:boxes`, a list of `%{xmin: x, ymin: y, xmax: x, ymax: y}`, and `:labels`, the label of each box.

    * `"ocr"` - `:quad_boxes`, a list of `[x1, y1, x2, y2, x3, y3, x4, y4]`, and `:labels`, the text of each quad box.

    * `"polygons"` - `:polygons`, a list with the polygons of each label, and `:labels`. A polygon is `[x1, y1, x2, y2, ...]`.

    * `"description_with_bboxes_or_polygons"` - `:boxes`, `:box_labels`, `:polygons` and `:polygon_labels`.
  """
  def parse(processor, task, text, size) do
    clean = String.replace(text, ["<s>", "</s>", "<pad>"], "")
    result = %{text: text |> String.replace(["<s>", "</s>"], "") |> String.trim()}

    case processor.types[task] || "pure_text" do
      "pure_text" ->
        result

      "description_with_bboxes" ->
        Map.merge(result, boxes(clean, size, false, MapSet.new()))

      "bboxes" ->
        Map.merge(result, boxes(clean, size, true, MapSet.new()))

      "phrase_grounding" ->
        Map.merge(result, boxes(clean, size, false, @banned_grounding_tokens))

      "ocr" ->
        Map.merge(result, quad_boxes(clean, size))

      "polygons" ->
        Map.merge(result, polygons(clean, size, true))

      "description_with_bboxes_or_polygons" ->
        empty = %{boxes: [], box_labels: [], polygons: [], polygon_labels: []}

        if String.contains?(clean, "<poly>") do
          %{polygons: polygons, labels: labels} = polygons(clean, size, false)
          Map.merge(result, %{empty | polygons: polygons, polygon_labels: labels})
        else
          %{boxes: boxes, labels: labels} = boxes(clean, size, false, MapSet.new())
          Map.merge(result, %{empty | boxes: boxes, box_labels: labels})
        end

      type ->
        raise ArgumentError, "unknown post-processing type #{inspect(type)} of the task #{task}"
    end
  end

  @phrase ~r/^\s*(.*?)(?=<od>|<\/od>|<box>|<\/box>|<bbox>|<\/bbox>|<loc_)/u
  @box ~r/<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/u

  # parse_description_with_bboxes_from_text_and_spans and
  # parse_phrase_grounding_from_text_and_spans: each phrase is a text and four or more
  # location tokens, and each group of four is a box with the label of the phrase.
  defp boxes(text, size, empty?, banned) do
    pattern =
      if empty?, do: ~r/(?:(?:<loc_\d+>){4,})/u, else: ~r/([^<]+(?:<loc_\d+>){4,})/u

    instances =
      for [phrase_text | _] <- Regex.scan(pattern, text),
          phrase_text = phrase_text |> replace_once("<ground>") |> replace_once("<obj>"),
          phrase_text != "" or empty?,
          [phrase | _] <- [Regex.run(@phrase, phrase_text)],
          phrase = String.trim(phrase),
          phrase not in banned,
          [_ | bins] <- Regex.scan(@box, phrase_text) do
        [x1, y1, x2, y2] = dequantize(bins, size)
        {%{xmin: x1, ymin: y1, xmax: x2, ymax: y2}, ascii(phrase)}
      end

    %{boxes: Enum.map(instances, &elem(&1, 0)), labels: Enum.map(instances, &elem(&1, 1))}
  end

  @quad ~r/(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/u

  # parse_ocr_from_text_and_spans: a text and the 4 corners of its quad box.
  defp quad_boxes(text, size) do
    instances =
      for [_match, content | bins] <- Regex.scan(@quad, text),
          do: {dequantize(bins, size), String.trim(content)}

    %{quad_boxes: Enum.map(instances, &elem(&1, 0)), labels: Enum.map(instances, &elem(&1, 1))}
  end

  @polygon_phrase ~r/^\s*(.*?)(?=<od>|<\/od>|<box>|<\/box>|<bbox>|<\/bbox>|<loc_|<poly>)/u
  @polygon_instance ~r/<poly>(.*?)<\/poly>/u
  @polygon ~r/((?:<loc_\d+>)+)(?:<sep>|$)/u
  @location ~r/<loc_(\d+)>/u

  # parse_description_with_polygons_from_text_and_spans: the polygons of a phrase are
  # separated by <sep>, and <poly> and </poly> enclose the polygons of one instance.
  defp polygons(text, size, empty?) do
    pattern =
      if empty?,
        do: ~r/(?:(?:<loc_\d+>|<sep>|<poly>|<\/poly>){4,})/u,
        else: ~r/([^<]+(?:<loc_\d+>|<sep>|<poly>|<\/poly>){4,})/u

    instances =
      for [phrase_text | _] <- Regex.scan(pattern, text),
          stripped = Regex.replace(~r/^<loc_\d+>/u, phrase_text, "", global: false),
          stripped != "" or empty?,
          [phrase | _] <- [Regex.run(@polygon_phrase, stripped)],
          instance <- polygon_instances(phrase_text),
          polygons = polygon_lists(instance, size),
          polygons != [] do
        {polygons, String.trim(phrase)}
      end

    %{polygons: Enum.map(instances, &elem(&1, 0)), labels: Enum.map(instances, &elem(&1, 1))}
  end

  defp polygon_instances(phrase_text) do
    if String.contains?(phrase_text, "<poly>") and String.contains?(phrase_text, "</poly>"),
      do: for([_, instance] <- Regex.scan(@polygon_instance, phrase_text), do: instance),
      else: [phrase_text]
  end

  # An odd number of coordinates drops the last one.
  defp polygon_lists(instance, size) do
    for [_, locations] <- Regex.scan(@polygon, instance) do
      bins = for [_, bin] <- Regex.scan(@location, locations), do: bin
      bins |> Enum.take(length(bins) - rem(length(bins), 2)) |> dequantize(size)
    end
  end

  # The bins alternate between x and y.
  defp dequantize(bins, {height, width}) do
    bins
    |> Enum.with_index()
    |> Enum.map(fn {bin, i} ->
      side = if rem(i, 2) == 0, do: width, else: height
      (String.to_integer(bin) + 0.5) * (side / 1000)
    end)
  end

  defp replace_once(text, pattern), do: String.replace(text, pattern, "", global: false)

  # Python transformers keeps the ASCII characters of a label.
  defp ascii(text), do: for(<<byte <- text>>, byte < 128, into: "", do: <<byte>>)
end
