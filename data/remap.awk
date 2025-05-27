/^#/ {
  next
}

# Process data lines
{
  # For the first field (FromNodeId)
  if (!($1 in node_map)) {
    node_map[$1] = ++count
  }
  # For the second field (ToNodeId)
  if (!($2 in node_map)) {
    node_map[$2] = ++count
  }
  # Print the remapped values
  # Ensure to handle cases where a field might be empty (though not expected in this format)
  if ($1 != "" && $2 != "") {
    print node_map[$1] "\t" node_map[$2]
  } else if ($1 != "") {
    print node_map[$1]
  } else {
    print # Should not happen with the provided format
  }
}
